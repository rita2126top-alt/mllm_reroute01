"""Isolated evaluation and exact release/GC matrix enumeration."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from .config import (ROOT, MODELS, TIERS, load_experiment, original_config_names,
                     gc_config_names, resolve_tasks)


@dataclass(frozen=True)
class MatrixJob:
    config: str
    arm: str
    checkpoint: str | None
    ablation: str
    tasks: str

    @property
    def key(self):
        return self.arm + "__" + self.ablation + "__" + self.config.replace("/", "__")


def build_matrix(*, models=MODELS, tiers=TIERS, arms="all", checkpoint_dir=None,
                 ablation="full", tasks="all") -> list[MatrixJob]:
    if arms not in ("all", "original", "gc"):
        raise ValueError("arms must be all, original, or gc")
    if any(model not in MODELS for model in models) or any(tier not in TIERS for tier in tiers):
        raise ValueError("Unknown model or tier")
    resolve_tasks(tasks)
    jobs = []
    if arms != "gc":
        jobs.extend(MatrixJob(cfg, "original", None, "full", tasks)
                    for cfg in original_config_names(models, tiers))
    if arms != "original":
        if checkpoint_dir is None:
            raise ValueError("A checkpoint directory is required for GC matrix jobs")
        for cfg in gc_config_names(models, tiers):
            spec = load_experiment(cfg)
            checkpoint = Path(checkpoint_dir).resolve() / f"{spec.checkpoint_key}__{ablation}.pt"
            jobs.append(MatrixJob(cfg, "gc", str(checkpoint), ablation, tasks))
    return jobs


@contextmanager
def release_eval_environment():
    """Restore presets to prevent one in-process run contaminating another model."""
    keys = ("REFCOCO_PROMPT_STYLE", "BBOX_COORD_FORMAT")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_evaluation(config, *, tasks="all", checkpoint=None, ablation="full", limit=None):
    """Reuse the release lmms-eval loader, task defaults and grounding presets."""
    spec = load_experiment(config)
    if checkpoint is not None:
        from .loading import validate_checkpoint
        validate_checkpoint(spec, checkpoint, ablation)
    from scripts import run_eval as release
    cfg = spec.as_omegaconf()
    cfg.eval.benchmarks = resolve_tasks(tasks)
    cfg.eval.batch_size = 1
    cfg.eval.log_samples = True
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        cfg.eval.limit = limit
    with release_eval_environment():
        if checkpoint is None:
            return release.run_lmms_eval(cfg)
        from .loading import patch_experiment
        def install_gc(model, **kwargs):
            return patch_experiment(model, spec, checkpoint, ablation)
        # The original run function receives its original config and loader. Only
        # its routing installation call is replaced, within this isolated call.
        with patch.object(release, "patch_model_for_routing", install_gc):
            return release.run_lmms_eval(cfg)


def new_output_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    forbidden = [(ROOT / "experiments" / name).resolve()
                 for name in ("logs", "profile", "runtime")]
    for directory in forbidden:
        if output == directory or directory in output.parents:
            raise ValueError("Use a separate Cold/Ghost output directory, outside release results")
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str)


def run_matrix(jobs: list[MatrixJob], output: Path, *, limit=None, dry_run=False,
               runner=subprocess.run, mode="evaluate", n_passes=None,
               n_warmup_passes=None, n_decode_tokens=None) -> int:
    """Each experiment is a separate process; preserve all failures and exit nonzero."""
    if mode not in ("evaluate", "profile", "benchmark"):
        raise ValueError("mode must be evaluate, profile, or benchmark")
    if mode != "evaluate" and limit is not None:
        raise ValueError("Sample limits are only supported for accuracy evaluation")
    output = Path(output).resolve()
    plan = []
    for job in jobs:
        command = [sys.executable, str(ROOT / "scripts" / "cold_ghost" / f"{mode}.py"),
                   "--config", job.config,
                   "--out", str(output / job.key), "--ablation", job.ablation]
        if mode == "evaluate":
            command.extend(["--tasks", job.tasks])
        else:
            for flag, value in (("--n-passes", n_passes), ("--n-warmup-passes", n_warmup_passes)):
                if value is not None:
                    command.extend([flag, str(value)])
            if mode == "benchmark" and n_decode_tokens is not None:
                command.extend(["--n-decode-tokens", str(n_decode_tokens)])
        if job.checkpoint:
            command.extend(["--checkpoint", job.checkpoint])
        else:
            command.append("--original")
        if limit is not None:
            command.extend(["--limit", str(limit)])
        plan.append({**asdict(job), "key": job.key, "mode": mode, "command": command})
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    output = new_output_dir(output)
    write_json(output / "plan.json", plan)
    records = []
    for item in plan:
        with (output / f"{item['key']}.log").open("x", encoding="utf-8") as logfile:
            try:
                result = runner(item["command"], stdout=logfile, stderr=subprocess.STDOUT,
                                cwd=ROOT, check=False)
                code = result.returncode
            except OSError as exc:
                logfile.write(str(exc))
                code = 127
        record = {"key": item["key"], "exit_code": code}
        result_path = output / item["key"] / "results.json"
        if code == 0 and result_path.exists():
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                metrics = data.get("results", data.get("aggregate", {}))
                if not metrics:
                    raise ValueError("results.json contains no metrics")
                record["results"] = metrics
            except (OSError, ValueError, AttributeError) as exc:
                record["exit_code"] = 1
                record["error"] = f"Invalid result file: {exc}"
        elif code == 0:
            record["exit_code"] = 1
            record["error"] = "Process succeeded without results.json"
        records.append(record)
    write_json(output / "summary.json", {"runs": records,
        "failed": sum(record["exit_code"] != 0 for record in records)})
    return int(any(record["exit_code"] != 0 for record in records))
