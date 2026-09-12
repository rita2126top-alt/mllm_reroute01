import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cold_ghost import full_suite as suite
from cold_ghost.config import resolve_tasks


def test_fixed_protocol_job_counts_and_commands_without_data(tmp_path):
    jobs = suite.build_plan(tmp_path / "not_yet_trained")
    payload = suite.plan_payload(jobs, manifest=tmp_path / "missing.json", images=tmp_path / "images",
                                 checkpoint_dir=tmp_path / "not_yet_trained")
    assert len(jobs) == 247
    assert payload["stage_counts"] == dict(train=1, accuracy=62, profile=62, benchmark=62, ablation=48, diagnostics=12)
    assert len(payload["accuracy_tasks"]) == 12
    assert len(payload["ablation_tasks"]) == 4
    for row in payload["plan"]:
        command = row["command"]
        assert "--limit" not in command and "--smoke" not in command
        if row["stage"] == "train":
            assert command[command.index("--ablation") + 1] == "all"
        if row["stage"] == "ablation":
            assert "stagewise" not in row["config"]
            assert row["ablation"] != "full"
        if row["stage"] == "benchmark":
            assert command[command.index("--n-passes") + 1] == "5"
            assert command[command.index("--n-warmup-passes") + 1] == "2"
            assert command[command.index("--n-decode-tokens") + 1] == "64"
    assert not tmp_path.joinpath("not_yet_trained").exists()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"manifest_sha256":"' + "0" * 64 + '"}')
    images, checkpoints = tmp_path / "images", tmp_path / "checkpoints"
    images.mkdir()
    checkpoints.mkdir()
    monkeypatch.setattr(suite, "_sources_sha256", lambda: "sources-v1")
    monkeypatch.setattr(suite, "_runtime_identity", lambda device: {"gpu_models": ["test-fixture"]})
    monkeypatch.setattr(suite, "_judge_settings", lambda: {"model": "gpt-4o", "key_configured": True})
    monkeypatch.setattr(suite, "_assert_judge_log", lambda path: None)
    return dict(manifest=manifest, images=images, checkpoint_dir=checkpoints, out=tmp_path / "output",
                manifest_loader=lambda *a, **k: {"manifest_sha256": "0" * 64},
                validator=lambda *a: None)


def write_accuracy(command, value=0.5):
    artifacts = Path(command[command.index("--out") + 1])
    artifacts.mkdir(parents=True, exist_ok=False)
    tasks = resolve_tasks(command[command.index("--tasks") + 1])
    config = command[command.index("--config") + 1]
    ablation = command[command.index("--ablation") + 1]
    checkpoint = command[command.index("--checkpoint") + 1] if "--checkpoint" in command else None
    (artifacts / "run.json").write_text(json.dumps(dict(config=config, scope="formal", limit=None,
                                                        tasks=tasks, ablation=ablation, checkpoint=checkpoint)))
    (artifacts / "results.json").write_text(json.dumps({"results": {t: {"acc": value} for t in tasks},
                    "n-samples": {t: {"original": 10, "effective": 10} for t in tasks}}))


def selected_jobs(environment, count=2, gc=False):
    return [job for job in suite.build_plan(environment["checkpoint_dir"])
            if job.stage == "accuracy" and (job.arm == "gc") == gc][:count]


def test_serial_failure_retry_new_attempt_and_verified_resume(environment, monkeypatch):
    jobs = selected_jobs(environment)
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    calls = []
    fail_once = {jobs[0].config}
    def runner(command, **kwargs):
        calls.append(command)
        config = command[command.index("--config") + 1]
        if config in fail_once:
            fail_once.remove(config)
            return SimpleNamespace(returncode=7)
        write_accuracy(command)
        return SimpleNamespace(returncode=0)
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 1
    assert len(calls) == 2  # Failure did not prevent the second independent job.
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 3  # The valid second job was skipped.
    failed_dir = environment["out"] / "accuracy" / jobs[0].key
    assert (failed_dir / "attempt-0001/failure.json").exists()
    assert (failed_dir / "attempt-0002/completion.json").exists()
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 3


def test_resume_rechecks_results_checkpoint_bytes_and_manifest_content(environment, monkeypatch):
    jobs = selected_jobs(environment, count=1, gc=True)
    Path(jobs[0].checkpoint).write_bytes(b"checkpoint-version-one")
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        write_accuracy(command)
        return SimpleNamespace(returncode=0)
    assert suite.run_suite(stage="accuracy", runner=runner, **environment) == 0
    attempt = environment["out"] / "accuracy" / jobs[0].key / "attempt-0001"
    data = json.loads((attempt / "artifacts/results.json").read_text())
    data["results"][resolve_tasks("all")[0]]["acc"] = 0.9
    (attempt / "artifacts/results.json").write_text(json.dumps(data))
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 2
    Path(jobs[0].checkpoint).write_bytes(b"checkpoint-version-two")
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 3
    environment["manifest"].write_text(environment["manifest"].read_text() + "\n")
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 4
    assert (attempt / "artifacts/results.json").exists()  # No old result was deleted.


def test_config_content_identity_invalidates_completed_attempt(environment, monkeypatch):
    jobs = selected_jobs(environment, count=1)
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    revision = {"value": "config-bytes-one"}
    monkeypatch.setattr(suite, "_config_identity", lambda name: {"name": name, "yaml_sha256": revision["value"]})
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        write_accuracy(command)
        return SimpleNamespace(returncode=0)
    suite.run_suite(stage="accuracy", runner=runner, **environment)
    revision["value"] = "config-bytes-two"
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert len(calls) == 2


def test_all_blocks_dependent_stages_after_training_failure(environment, monkeypatch):
    complete_plan = suite.build_plan(environment["checkpoint_dir"])
    jobs = [complete_plan[0]] + selected_jobs(environment, count=2)
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    assert suite.run_suite(stage="all", runner=runner, **environment) == 1
    assert len(calls) == 1
    summary = json.loads((environment["out"] / "summary.json").read_text())
    assert summary["failed"] == 1 and summary["blocked"] == 2


def test_independent_accuracy_does_not_require_training_stage_receipt(environment, monkeypatch):
    jobs = selected_jobs(environment, count=1, gc=True)
    Path(jobs[0].checkpoint).write_bytes(b"externally-completed-checkpoint")
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    def runner(command, **kwargs):
        write_accuracy(command)
        return SimpleNamespace(returncode=0)
    assert suite.run_suite(stage="accuracy", runner=runner, resume=True, **environment) == 0
    assert not (environment["out"] / "train").exists()


def test_protocol_rejects_partial_task_counts(environment, tmp_path):
    job = selected_jobs(environment, count=1)[0]
    artifacts = tmp_path / "artifacts"
    command = suite.job_command(job, artifacts, manifest=environment["manifest"], images=environment["images"],
                               checkpoint_dir=environment["checkpoint_dir"])
    write_accuracy(command)
    path = artifacts / "results.json"
    data = json.loads(path.read_text())
    data["n-samples"][resolve_tasks("all")[0]]["effective"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="every sample"):
        suite.validate_artifacts(job, artifacts, manifest_sha256="0" * 64,
                                 checkpoint_dir=environment["checkpoint_dir"])


def test_missing_judge_preflight_stops_before_training(environment, monkeypatch):
    monkeypatch.setattr(suite, "_judge_settings", lambda: (_ for _ in ()).throw(ValueError("judge key absent")))
    with pytest.raises(ValueError, match="judge key absent"):
        suite.run_suite(stage="all", **environment)
    assert not environment["out"].exists()


def test_judge_failure_log_prevents_false_success(environment, monkeypatch):
    jobs = selected_jobs(environment, count=1)
    monkeypatch.setattr(suite, "build_plan", lambda directory: jobs)
    monkeypatch.setattr(suite, "_assert_judge_log", lambda path: (_ for _ in ()).throw(ValueError("judge fallback")))
    def runner(command, **kwargs):
        write_accuracy(command)
        return SimpleNamespace(returncode=0)
    assert suite.run_suite(stage="accuracy", runner=runner, **environment) == 1
    summary = json.loads((environment["out"] / "summary.json").read_text())
    assert summary["failed"] == 1


def test_device_selection_maps_existing_visible_gpu_indices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert suite.child_environment("cuda:1")["CUDA_VISIBLE_DEVICES"] == "5"
    with pytest.raises(ValueError):
        suite.child_environment("cpu")


def test_output_lock_excludes_other_runners_and_recovers_after_release(tmp_path):
    out = tmp_path / "suite"
    with suite._suite_lock(out):
        with pytest.raises(RuntimeError, match="Another full-suite runner"):
            with suite._suite_lock(out):
                pass
    # The owner file remains, but the OS lock is released and resume is safe.
    assert out.with_name(out.name + ".full-suite.lock").is_file()
    with suite._suite_lock(out):
        pass


def test_efficiency_checks_actual_cohort_ids_and_exact_passes(environment, tmp_path):
    job = next(job for job in suite.build_plan(environment["checkpoint_dir"]) if job.stage == "profile")
    artifacts = tmp_path / "profile-artifacts"
    artifacts.mkdir()
    cohort = json.loads((suite.ROOT / "bench_data/manifest.json").read_text(encoding="utf-8"))
    data = {"config": job.config, "mode": "profile", "ablation": "full", "checkpoint": None,
            "settings": {"n_passes": 1, "n_warmup_passes": 0, "n_decode_tokens": None},
            "per_sample": [{"sample_id": row["sample_id"], "pass": 0} for row in cohort],
            "aggregate": {"kv_bytes": {"n": 3}, "model_registered_flops": {"n": 3}}}
    path = artifacts / "results.json"
    path.write_text(json.dumps(data))
    assert suite.validate_artifacts(job, artifacts, manifest_sha256="0" * 64,
                                    checkpoint_dir=environment["checkpoint_dir"])
    data["per_sample"][0]["sample_id"] = "unknown-image"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown"):
        suite.validate_artifacts(job, artifacts, manifest_sha256="0" * 64,
                                 checkpoint_dir=environment["checkpoint_dir"])
