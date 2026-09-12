#!/usr/bin/env python
"""Serial formal training of the 12 original compact Reroute configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)

from cold_ghost.config import ABLATIONS, MODELS, TIERS, ROOT, gc_config_names, load_experiment


# The two release backbones are fixed 7B architectures. Check these dimensions
# without downloading or instantiating a backbone just to inspect a checkpoint.
HIDDEN_SIZES = {"llava": 4096, "qwen25vl": 3584}


def plan_jobs(models, tiers, ablations, *, manifest, images, output_dir, device):
    """One checkpoint serves both compact and stagewise execution."""
    jobs = []
    for config_name in gc_config_names(models=models, tiers=tiers):
        spec = load_experiment(config_name)
        if spec.routing["action"] != "compact_route":
            continue
        for ablation in ablations:
            output = Path(output_dir) / f"{spec.checkpoint_key}__{ablation}.pt"
            command = [sys.executable, str(ROOT / "scripts/cold_ghost/train.py"),
                       "--config", spec.config_name, "--manifest", str(manifest),
                       "--images", str(images), "--out", str(output),
                       "--ablation", ablation, "--device", device]
            jobs.append({"config": spec.config_name, "ablation": ablation,
                         "checkpoint": str(output), "command": command})
    return jobs


def checkpoint_state(path, spec, ablation, manifest_sha256):
    """Return new/complete/partial after strict Ghost-only checkpoint checks.

    A partial checkpoint must additionally restore into the exact formal
    optimizer/training settings. Invalid checkpoints are never overwritten.
    """
    path = Path(path)
    if not path.exists():
        return "new"
    from cold_ghost.checkpoint import experiment_metadata, load_checkpoint
    from cold_ghost.ghost import GhostConfig, GhostModules
    from cold_ghost.training import TwoStageTrainer

    modules = GhostModules(HIDDEN_SIZES[spec.model_family], spec.model["num_layers"],
                           GhostConfig(ablation=ablation))
    expected = experiment_metadata(spec, modules, manifest_sha256)
    payload = load_checkpoint(path, modules, expected_metadata=expected, require_complete=False)
    state = payload.get("training_state", {})
    if state.get("smoke", False):
        raise ValueError("Training matrix refuses smoke checkpoints")
    if state.get("complete", False):
        # This is the same formal-completion rule used by the evaluation loader.
        load_checkpoint(path, modules, expected_metadata=expected, require_complete=True)
        return "complete"
    trainer = TwoStageTrainer(modules, expected, path, smoke=False)
    trainer.resume(path)
    return "partial"


def write_summary(path, results, planned_jobs):
    summary = {"planned_jobs": planned_jobs, "finished_jobs": len(results),
               "trained": sum(row["status"] == "trained" for row in results),
               "skipped_complete": sum(row["status"] == "skipped_complete" for row in results),
               "failed": sum(row["status"] == "failed" for row in results),
               "interrupted": any(row["status"] == "interrupted" for row in results),
               "results": results}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return summary


def run_jobs(jobs, manifest_sha256, summary_path, *, runner=None, validator=None):
    """Run exactly one child at a time; preserve failures and continue jobs."""
    runner = subprocess.run if runner is None else runner
    validator = checkpoint_state if validator is None else validator
    results = []
    for index, job in enumerate(jobs):
        result = {key: job[key] for key in ("config", "ablation", "checkpoint")}
        try:
            spec = load_experiment(job["config"])
            state = validator(job["checkpoint"], spec, job["ablation"], manifest_sha256)
            if state == "complete":
                result["status"] = "skipped_complete"
                print(json.dumps(result), flush=True)
            else:
                command = list(job["command"])
                if state == "partial":
                    command.extend(["--resume", job["checkpoint"]])
                elif state != "new":
                    raise ValueError(f"Unexpected checkpoint state: {state}")
                result["resumed"] = state == "partial"
                result["command"] = command
                print(json.dumps({"job": index + 1, "jobs": len(jobs), **result}), flush=True)
                child = runner(command, cwd=str(ROOT), check=False)
                result["returncode"] = child.returncode
                if child.returncode != 0:
                    raise RuntimeError(f"Training subprocess exited with code {child.returncode}")
                if validator(job["checkpoint"], spec, job["ablation"], manifest_sha256) != "complete":
                    raise RuntimeError("Training exited successfully without a matching complete formal checkpoint")
                result["status"] = "trained"
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Interrupted by user")
            results.append(result)
            write_summary(summary_path, results, len(jobs))
            return 130
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(json.dumps(result), file=sys.stderr, flush=True)
        results.append(result)
        write_summary(summary_path, results, len(jobs))
    summary = write_summary(summary_path, results, len(jobs))
    print(json.dumps({"summary": str(summary_path), **{key: summary[key] for key in
                      ("planned_jobs", "trained", "skipped_complete", "failed")}}), flush=True)
    return 1 if summary["failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all",) + MODELS, default="all")
    parser.add_argument("--tier", choices=("all",) + TIERS, default="all")
    parser.add_argument("--ablation", choices=("all",) + ABLATIONS, default="full")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cold_ghost")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without loading data, checkpoints, models, or starting training")
    args = parser.parse_args(argv)
    models = MODELS if args.model == "all" else (args.model,)
    tiers = TIERS if args.tier == "all" else (args.tier,)
    ablations = ABLATIONS if args.ablation == "all" else (args.ablation,)
    manifest_path, images_path = Path(args.manifest).resolve(), Path(args.images).resolve()
    output_dir = Path(args.output_dir).resolve()
    jobs = plan_jobs(models, tiers, ablations, manifest=manifest_path, images=images_path,
                     output_dir=output_dir, device=args.device)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "jobs": len(jobs),
                          "existing_checkpoint_policy": "validate, skip complete, resume matching partial; smoke rejected",
                          "checkpoint_status": "not inspected during dry run", "plan": jobs}, indent=2))
        return 0
    from cold_ghost.data import load_manifest
    try:
        manifest = load_manifest(manifest_path, allow_smoke=False)
        if not images_path.is_dir():
            raise ValueError(f"Image directory does not exist: {images_path}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return run_jobs(jobs, manifest["manifest_sha256"], output_dir / "train_matrix_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
