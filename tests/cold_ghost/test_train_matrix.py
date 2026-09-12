import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cold_ghost.config import ROOT, MODELS, TIERS, ABLATIONS


def matrix_module():
    spec = importlib.util.spec_from_file_location("gc_train_matrix", ROOT / "scripts/cold_ghost/train_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jobs(module, tmp_path, ablations=("full",), models=MODELS, tiers=TIERS):
    return module.plan_jobs(models, tiers, ablations, manifest=tmp_path / "manifest.json",
                            images=tmp_path / "images", output_dir=tmp_path, device="cuda")


def test_exact_twelve_or_sixty_unique_compact_checkpoints(tmp_path):
    module = matrix_module()
    full = jobs(module, tmp_path)
    all_ablations = jobs(module, tmp_path, ABLATIONS)
    assert len(full) == 12
    assert len(all_ablations) == 60
    assert len({job["checkpoint"] for job in all_ablations}) == 60
    assert all("stagewise" not in job["config"] for job in all_ablations)
    assert all("--smoke" not in job["command"] for job in all_ablations)


def test_dry_run_needs_no_data_and_rejects_smoke_option(tmp_path, capsys):
    module = matrix_module()
    args = ["--model", "qwen25vl", "--tier", "avg64", "--ablation", "all",
            "--manifest", str(tmp_path / "missing.json"), "--images", str(tmp_path / "missing"),
            "--output-dir", str(tmp_path / "output"), "--dry-run"]
    assert module.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["jobs"] == 10
    assert not (tmp_path / "output").exists()
    with pytest.raises(SystemExit) as exc:
        module.main(args + ["--smoke"])
    assert exc.value.code == 2


def test_serial_skip_resume_failure_continues_and_summary(tmp_path):
    module = matrix_module()
    planned = jobs(module, tmp_path)[:4]
    states = {planned[0]["checkpoint"]: "complete", planned[1]["checkpoint"]: "partial",
              planned[2]["checkpoint"]: "new", planned[3]["checkpoint"]: "new"}
    commands = []
    def validator(path, *_):
        return states[path]
    def runner(command, **kwargs):
        assert kwargs["check"] is False
        commands.append(command)
        path = command[command.index("--out") + 1]
        if path == planned[2]["checkpoint"]:
            return SimpleNamespace(returncode=3)
        states[path] = "complete"
        return SimpleNamespace(returncode=0)
    summary_path = tmp_path / "summary.json"
    assert module.run_jobs(planned, "0" * 64, summary_path, runner=runner, validator=validator) == 1
    assert len(commands) == 3
    assert commands[0][-2:] == ["--resume", planned[1]["checkpoint"]]
    assert "--resume" not in commands[2]
    summary = json.loads(summary_path.read_text())
    assert summary["skipped_complete"] == 1
    assert summary["trained"] == 2
    assert summary["failed"] == 1


def test_success_without_complete_checkpoint_is_failure(tmp_path):
    module = matrix_module()
    planned = jobs(module, tmp_path)[:1]
    assert module.run_jobs(planned, "0" * 64, tmp_path / "summary.json",
                           runner=lambda *a, **k: SimpleNamespace(returncode=0),
                           validator=lambda *a: "new") == 1


def test_checkpoint_validator_strict_identity_resume_and_smoke(tmp_path, monkeypatch):
    from cold_ghost.checkpoint import experiment_metadata
    from cold_ghost.ghost import GhostConfig, GhostModules
    from cold_ghost.training import TwoStageTrainer
    module = matrix_module()
    monkeypatch.setitem(module.HIDDEN_SIZES, "llava", 16)
    spec = SimpleNamespace(model_family="llava", model={"num_layers": 5},
                           checkpoint_metadata=lambda: dict(backbone="tiny", model_family="llava",
                               num_layers=5, drop_layers=[1, 3], keep_ratios=[0.5, 0.5], checkpoint_key="tiny"))
    models = GhostModules(16, 5, GhostConfig())
    metadata = experiment_metadata(spec, models, "0" * 64)
    checkpoint = tmp_path / "formal.pt"
    trainer = TwoStageTrainer(models, metadata, checkpoint)
    trainer.save()
    assert module.checkpoint_state(checkpoint, spec, "full", "0" * 64) == "partial"
    trainer.state["complete"] = True
    trainer.save()
    assert module.checkpoint_state(checkpoint, spec, "full", "0" * 64) == "complete"
    with pytest.raises(ValueError, match="manifest_sha256"):
        module.checkpoint_state(checkpoint, spec, "full", "1" * 64)
    trainer.state["smoke"] = True
    trainer.save()
    with pytest.raises(ValueError, match="smoke"):
        module.checkpoint_state(checkpoint, spec, "full", "0" * 64)
