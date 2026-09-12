"""Fixed full-data experiment protocol with content-verified serial resumes.

Plans do not load checkpoints or data. Execution accepts only the formal
8192/512 manifest and launches the existing, separate experiment CLIs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from .config import ABLATIONS, ROOT, gc_config_names, load_experiment, resolve_tasks
from .evaluation import build_matrix, new_output_dir

STAGES = ("train", "accuracy", "profile", "benchmark", "ablation", "diagnostics")
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SuiteJob:
    stage: str
    key: str
    config: str | None = None
    arm: str = "original"
    ablation: str = "full"
    checkpoint: str | None = None
    self_checkpoint: str | None = None


def compact_configs():
    return [name for name in gc_config_names()
            if load_experiment(name).routing["action"] == "compact_route"]


def build_plan(checkpoint_dir) -> list[SuiteJob]:
    directory = Path(checkpoint_dir).resolve()
    jobs = [SuiteJob("train", "all_60_checkpoints", arm="gc")]
    matrix = build_matrix(checkpoint_dir=directory, arms="all", ablation="full", tasks="all")
    for stage in ("accuracy", "profile", "benchmark"):
        jobs.extend(SuiteJob(stage, row.key, row.config, row.arm, row.ablation, row.checkpoint)
                    for row in matrix)
    for ablation in ABLATIONS:
        if ablation == "full":
            continue
        for row in build_matrix(checkpoint_dir=directory, arms="gc", ablation=ablation, tasks="ablation"):
            if load_experiment(row.config).routing["action"] == "compact_route":
                jobs.append(SuiteJob("ablation", row.key, row.config, row.arm, ablation, row.checkpoint))
    for name in compact_configs():
        key = load_experiment(name).checkpoint_key
        jobs.append(SuiteJob("diagnostics", key, name, "gc", "full",
                             str(directory / f"{key}__full.pt"), str(directory / f"{key}__self.pt")))
    return jobs


def job_command(job, artifacts, *, manifest, images, checkpoint_dir):
    base = [sys.executable, str(ROOT / "scripts/cold_ghost")]
    artifacts = Path(artifacts)
    if job.stage == "train":
        return [base[0], str(Path(base[1]) / "train_matrix.py"), "--model", "all", "--tier", "all",
                "--ablation", "all", "--manifest", str(manifest), "--images", str(images),
                "--output-dir", str(checkpoint_dir), "--device", "cuda"]
    if job.stage == "diagnostics":
        return [base[0], str(Path(base[1]) / "diagnose.py"), "--config", job.config,
                "--manifest", str(manifest), "--images", str(images), "--checkpoint", job.checkpoint,
                "--self-checkpoint", job.self_checkpoint, "--device", "cuda",
                "--out", str(artifacts / "results.json")]
    script = "evaluate" if job.stage in ("accuracy", "ablation") else job.stage
    command = [base[0], str(Path(base[1]) / f"{script}.py"), "--config", job.config,
               "--out", str(artifacts), "--ablation", job.ablation]
    command.extend(["--checkpoint", job.checkpoint] if job.checkpoint else ["--original"])
    if job.stage in ("accuracy", "ablation"):
        command.extend(["--tasks", "all" if job.stage == "accuracy" else "ablation"])
    elif job.stage == "profile":
        command.extend(["--n-passes", "1", "--n-warmup-passes", "0"])
    else:
        command.extend(["--n-passes", "5", "--n-warmup-passes", "2", "--n-decode-tokens", "64"])
    return command


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sources_sha256():
    paths = set((ROOT / "cold_ghost").glob("*.py"))
    paths.update((ROOT / "scripts/cold_ghost").glob("*.py"))
    paths.update((ROOT / "models").glob("*.py"))
    paths.add(ROOT / "scripts/run_eval.py")
    return _digest({p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)})


def _config_identity(name):
    spec = load_experiment(name)
    return {"name": spec.config_name, "yaml_sha256": file_sha256(spec.path),
            "resolved_settings_sha256": _digest({"model": spec.model, "routing": spec.routing, "eval": spec.eval})}


def checkpoint_requirements(job, checkpoint_dir):
    if job.stage == "train":
        return [(name, ablation, str(Path(checkpoint_dir) / f"{load_experiment(name).checkpoint_key}__{ablation}.pt"))
                for name in compact_configs() for ablation in ABLATIONS]
    if job.checkpoint is None:
        return []
    requirements = [(job.config, job.ablation, job.checkpoint)]
    if job.self_checkpoint:
        requirements.append((job.config, "self", job.self_checkpoint))
    return requirements


def _validate_checkpoint(name, ablation, path, manifest_sha256):
    from .loading import validate_checkpoint
    payload = validate_checkpoint(load_experiment(name), path, ablation)
    if payload["metadata"]["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"Checkpoint training manifest mismatch: {path}")


def _judge_settings():
    from .judge import validate_mmbench_judge_environment
    return validate_mmbench_judge_environment()


def _assert_judge_log(path):
    from .judge import scan_mmbench_judge_log
    failures = scan_mmbench_judge_log(path)
    if failures:
        raise ValueError("MMBench judge failure markers: " + ", ".join(sorted({item["code"] for item in failures})))


def _runtime_identity(device):
    import torch
    gpus = []
    if torch.cuda.is_available():
        indices = [int(device.split(":")[1])] if ":" in device else range(torch.cuda.device_count())
        gpus = [torch.cuda.get_device_name(index) for index in indices]
    return {"gpu_models": gpus, "cuda_version": torch.version.cuda}


def _cohort_identity():
    base = ROOT / "bench_data"
    manifest = base / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    return {"manifest": file_sha256(manifest),
            "images": {row["image"]: file_sha256(base / row["image"]) for row in rows}}


def job_identity(job, *, manifest, manifest_sha256, checkpoint_dir, device, validator=_validate_checkpoint):
    configs = compact_configs() if job.stage == "train" else [job.config]
    checkpoints = {}
    for name, ablation, path in checkpoint_requirements(job, checkpoint_dir):
        if job.stage != "train":
            validator(name, ablation, path, manifest_sha256)
        checkpoints[f"{load_experiment(name).checkpoint_key}__{ablation}"] = (
            file_sha256(path) if Path(path).is_file() else None)
        if job.stage != "train" and checkpoints[f"{load_experiment(name).checkpoint_key}__{ablation}"] is None:
            raise FileNotFoundError(path)
    versions = {}
    for distribution in ("torch", "transformers", "lmms-eval"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {"protocol_version": FORMAT_VERSION, "stage": job.stage, "key": job.key,
            "configs": [_config_identity(name) for name in configs],
            "manifest_file_sha256": file_sha256(manifest), "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoints, "source_sha256": _sources_sha256(),
            "device": device, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "software": versions, "runtime": _runtime_identity(device),
            "judge": _judge_settings() if job.stage in ("accuracy", "ablation") else None,
            "cohort": _cohort_identity() if job.stage in ("profile", "benchmark") else None}


def _assert_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)


def validate_artifacts(job, artifacts, *, manifest_sha256, checkpoint_dir, validator=_validate_checkpoint):
    """Check actual expected task/protocol coverage, not merely file existence."""
    artifacts = Path(artifacts)
    data = _read(artifacts / "results.json")
    _assert_finite(data)
    if job.stage == "train":
        expected = {(name, ablation) for name in compact_configs() for ablation in ABLATIONS}
        rows = data.get("results", [])
        if (data.get("planned_jobs") != 60 or data.get("finished_jobs") != 60 or data.get("failed") != 0
                or len(rows) != 60 or {(r.get("config"), r.get("ablation")) for r in rows} != expected
                or any(r.get("status") not in ("trained", "skipped_complete") for r in rows)):
            raise ValueError("Training matrix did not complete all 60 matching checkpoints")
        for name, ablation, path in checkpoint_requirements(job, checkpoint_dir):
            validator(name, ablation, path, manifest_sha256)
    elif job.stage in ("accuracy", "ablation"):
        run = _read(artifacts / "run.json")
        tasks = resolve_tasks("all" if job.stage == "accuracy" else "ablation")
        if (run.get("config") != job.config or run.get("scope") != "formal" or run.get("limit") is not None
                or set(run.get("tasks", [])) != set(tasks) or run.get("ablation") != job.ablation
                or run.get("checkpoint") != job.checkpoint):
            raise ValueError("Accuracy run metadata does not match the complete formal protocol")
        results = data.get("results", {})
        if not isinstance(results, dict) or any(not isinstance(results.get(task), dict) or not results[task] for task in tasks):
            raise ValueError("Accuracy results are missing one or more required tasks")
        counts = data.get("n-samples", {})
        for task in tasks:
            count = counts.get(task, {})
            original, effective = count.get("original"), count.get("effective")
            if (type(original) is not int or type(effective) is not int
                    or original <= 0 or effective != original):
                raise ValueError(f"Task {task} was not evaluated on every sample")
    elif job.stage in ("profile", "benchmark"):
        passes, warmup = (1, 0) if job.stage == "profile" else (5, 2)
        settings = data.get("settings", {})
        if (data.get("config") != job.config or data.get("mode") != job.stage or data.get("ablation") != job.ablation
                or settings.get("n_passes") != passes or settings.get("n_warmup_passes") != warmup
                or settings.get("n_decode_tokens") != (64 if job.stage == "benchmark" else None)
                or len(data.get("per_sample", [])) != 3 * passes or not isinstance(data.get("aggregate"), dict)
                or "kv_bytes" not in data["aggregate"]
                or ("prefill_ms" if job.stage == "benchmark" else "model_registered_flops") not in data["aggregate"]
                or data.get("checkpoint") != job.checkpoint):
            raise ValueError("Efficiency output does not cover the fixed cohort/repetition protocol")
        cohort = json.loads((ROOT / "bench_data/manifest.json").read_text(encoding="utf-8"))
        expected_ids = {row["sample_id"] for row in cohort}
        if len(expected_ids) != 3:
            raise ValueError("Original efficiency cohort must have exactly three distinct sample IDs")
        for repetition in range(passes):
            samples = [r for r in data["per_sample"] if r.get("pass") == repetition]
            if len(samples) != 3 or {r.get("sample_id") for r in samples} != expected_ids:
                raise ValueError("Efficiency output has missing, unknown, or repeated cohort samples")
    else:
        if (data.get("config") != job.config or data.get("manifest_sha256") != manifest_sha256
                or data.get("samples") != 512
                or not all(key in data for key in ("staleness_by_full_skip_age", "dense_residual_rank_reconstruction",
                                                   "deployed_residual_prediction", "future_reactivation"))
                or set(data.get("deployed_residual_prediction", {})) != {"full", "self"}
                or any(data["deployed_residual_prediction"][arm].get("samples") != 512 for arm in ("full", "self"))):
            raise ValueError("Diagnostics did not cover all 512 validation samples and required comparisons")
    return {p.relative_to(artifacts).as_posix(): file_sha256(p)
            for p in sorted(artifacts.rglob("*")) if p.is_file()}


def _completed_attempt(job_dir, job, identity, *, manifest_sha256, checkpoint_dir, validator):
    for attempt in sorted(job_dir.glob("attempt-*"), reverse=True):
        try:
            receipt = _read(attempt / "completion.json")
            if receipt.get("status") != "succeeded" or receipt.get("identity_sha256") != _digest(identity):
                continue
            if _digest(receipt.get("identity")) != receipt["identity_sha256"]:
                continue
            if job.stage in ("accuracy", "ablation"):
                _assert_judge_log(attempt / "process.log")
            current = validate_artifacts(job, attempt / "artifacts", manifest_sha256=manifest_sha256,
                                         checkpoint_dir=checkpoint_dir, validator=validator)
            if current == receipt.get("artifacts_sha256"):
                return attempt
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _new_attempt(job_dir):
    job_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.split("-")[1]) for p in job_dir.iterdir()
               if p.is_dir() and re.fullmatch(r"attempt-\d+", p.name)]
    target = job_dir / f"attempt-{max(numbers, default=0) + 1:04d}"
    target.mkdir()
    return target


def child_environment(device):
    if not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("The full formal suite requires a CUDA device")
    env = dict(os.environ)
    if ":" in device:
        index = int(device.split(":")[1])
        current = env.get("CUDA_VISIBLE_DEVICES")
        if current is not None:
            devices = [item.strip() for item in current.split(",") if item.strip()]
            if index >= len(devices):
                raise ValueError("Requested CUDA index is outside CUDA_VISIBLE_DEVICES")
            env["CUDA_VISIBLE_DEVICES"] = devices[index]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(index)
    return env


def plan_payload(jobs, *, manifest, images, checkpoint_dir):
    return {"protocol_version": FORMAT_VERSION, "jobs": len(jobs),
            "stage_counts": {stage: sum(job.stage == stage for job in jobs) for stage in STAGES},
            "training_checkpoints": 60, "accuracy_tasks": resolve_tasks("all"),
            "ablation_tasks": resolve_tasks("ablation"), "diagnostic_samples_per_job": 512,
            "plan": [{**asdict(job), "command": job_command(job, "<attempt-artifacts>", manifest=manifest,
                       images=images, checkpoint_dir=checkpoint_dir)} for job in jobs]}


def _run_suite_locked(*, stage="all", manifest, images, checkpoint_dir, out, device="cuda", resume=False,
              runner=None, validator=None, manifest_loader=None):
    if stage not in ("all",) + STAGES:
        raise ValueError(f"Unknown suite stage: {stage}")
    if manifest_loader is None:
        from .data import load_manifest
        manifest_loader = load_manifest
    if stage in ("all", "accuracy", "ablation"):
        _judge_settings()
    formal = manifest_loader(manifest, allow_smoke=False)
    manifest_id = formal["manifest_sha256"]
    env = child_environment(device)
    if stage in ("all", "train", "diagnostics") and not Path(images).is_dir():
        raise ValueError(f"Image directory does not exist: {images}")
    runner = subprocess.run if runner is None else runner
    validator = _validate_checkpoint if validator is None else validator
    out = Path(out).resolve()
    if out.exists():
        if not resume or not (out / "suite.json").is_file():
            raise FileExistsError("Use a new suite output directory, or --resume an existing suite directory")
        if _read(out / "suite.json").get("protocol_version") != FORMAT_VERSION:
            raise ValueError("Existing suite directory uses another protocol version")
    else:
        new_output_dir(out)
        _write(out / "suite.json", {"protocol_version": FORMAT_VERSION,
                                   "created_at": datetime.now(timezone.utc).isoformat()})
    jobs = build_plan(checkpoint_dir)
    plan = plan_payload(jobs, manifest=manifest, images=images, checkpoint_dir=checkpoint_dir)
    _write(out / "plans" / f"{_digest(plan)}.json", plan)
    _write(out / "plan.json", plan)
    invocation = _new_attempt(out / "invocations")
    records, train_failed = [], False

    def save_summary():
        summary = {"protocol_version": FORMAT_VERSION, "stage": stage,
                   "invocation": str(invocation), "records": records,
                   "succeeded": sum(r["status"] == "succeeded" for r in records),
                   "skipped_valid": sum(r["status"] == "skipped_valid" for r in records),
                   "failed": sum(r["status"] in ("failed", "interrupted") for r in records),
                   "blocked": sum(r["status"] == "blocked" for r in records)}
        _write(invocation / "summary.json", summary)
        _write(out / "summary.json", summary)
        return summary

    for job in jobs:
        if stage != "all" and stage != job.stage:
            continue
        record = {"stage": job.stage, "key": job.key}
        if stage == "all" and train_failed and job.stage != "train":
            record.update(status="blocked", error="Full training matrix failed; dependent suite stages were not launched")
            records.append(record)
            save_summary()
            continue
        job_dir = out / job.stage / job.key
        attempt = None
        try:
            identity = job_identity(job, manifest=manifest, manifest_sha256=manifest_id,
                                    checkpoint_dir=checkpoint_dir, device=device, validator=validator)
            previous = (_completed_attempt(job_dir, job, identity, manifest_sha256=manifest_id,
                                           checkpoint_dir=checkpoint_dir, validator=validator) if resume else None)
            if previous:
                record.update(status="skipped_valid", attempt=str(previous))
            else:
                attempt = _new_attempt(job_dir)
                command = job_command(job, attempt / "artifacts", manifest=manifest, images=images,
                                      checkpoint_dir=checkpoint_dir)
                _write(attempt / "run.json", {"job": asdict(job), "command": command, "identity_before": identity})
                print(json.dumps({"stage": job.stage, "job": job.key, "log": str(attempt / "process.log")}), flush=True)
                with (attempt / "process.log").open("x", encoding="utf-8") as logfile:
                    result = runner(command, cwd=str(ROOT), env=env, stdout=logfile,
                                    stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Subprocess exited with code {result.returncode}")
                if job.stage in ("accuracy", "ablation"):
                    _assert_judge_log(attempt / "process.log")
                if job.stage == "train":
                    _write(attempt / "artifacts/results.json", _read(Path(checkpoint_dir) / "train_matrix_summary.json"))
                hashes = validate_artifacts(job, attempt / "artifacts", manifest_sha256=manifest_id,
                                            checkpoint_dir=checkpoint_dir, validator=validator)
                # Training creates/changes checkpoints; bind completion to their
                # actual final bytes. Other jobs must not change their inputs.
                after = job_identity(job, manifest=manifest, manifest_sha256=manifest_id,
                                     checkpoint_dir=checkpoint_dir, device=device, validator=validator)
                before_inputs = {key: value for key, value in identity.items() if key != "checkpoint_sha256" or job.stage != "train"}
                after_inputs = {key: value for key, value in after.items() if key != "checkpoint_sha256" or job.stage != "train"}
                if after_inputs != before_inputs:
                    raise RuntimeError("Job inputs changed while the subprocess was running")
                _write(attempt / "completion.json", {"status": "succeeded", "identity": after,
                       "identity_sha256": _digest(after), "artifacts_sha256": hashes})
                record.update(status="succeeded", attempt=str(attempt))
        except KeyboardInterrupt:
            record.update(status="interrupted", error="Interrupted by user")
            if attempt is not None:
                _write(attempt / "failure.json", record)
            records.append(record)
            save_summary()
            return 130
        except Exception as exc:
            if attempt is None:
                attempt = _new_attempt(job_dir)
            record.update(status="failed", attempt=str(attempt), error=f"{type(exc).__name__}: {exc}")
            _write(attempt / "failure.json", record)
            print(json.dumps(record), file=sys.stderr, flush=True)
            train_failed = train_failed or job.stage == "train"
        records.append(record)
        save_summary()
    summary = save_summary()
    print(json.dumps({"summary": str(out / "summary.json"), "failed": summary["failed"],
                      "blocked": summary["blocked"], "skipped_valid": summary["skipped_valid"]}), flush=True)
    return int(bool(summary["failed"] or summary["blocked"]))


@contextmanager
def _suite_lock(out):
    """OS lock releases automatically after exit/crash; stale PID files are safe.

    Keep the inode/file in place: unlinking a lock file creates a race where
    another process could lock an old inode while a third locks the new one.
    The JSON owner record is informational; the operating system is authoritative.
    """
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.with_name(out.name + ".full-suite.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b" ")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(f"Another full-suite runner owns this output directory: {out}") from exc
        owner = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "out": str(out),
                            "started_at": datetime.now(timezone.utc).isoformat()}).encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, owner)
        os.ftruncate(descriptor, len(owner))
        os.fsync(descriptor)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_suite(*, stage="all", manifest, images, checkpoint_dir, out, device="cuda", resume=False,
              runner=None, validator=None, manifest_loader=None):
    with _suite_lock(out):
        return _run_suite_locked(stage=stage, manifest=manifest, images=images, checkpoint_dir=checkpoint_dir,
                                 out=out, device=device, resume=resume, runner=runner,
                                 validator=validator, manifest_loader=manifest_loader)
