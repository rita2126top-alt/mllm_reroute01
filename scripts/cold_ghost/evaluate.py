#!/usr/bin/env python3
"""Evaluate one unchanged release config with an original or trained GC arm."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)

from cold_ghost.config import ABLATIONS, load_experiment, resolve_tasks
from cold_ghost.evaluation import new_output_dir, run_evaluation, write_json
from cold_ghost.judge import mmbench_failure_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original release YAML path or experiment name")
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--checkpoint", type=Path, help="Complete trained Ghost checkpoint; never random weights")
    arm.add_argument("--original", action="store_true", help="Run unchanged original method")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--tasks", default="all", help="all/paper_main/vqa/grounding/ablation or CSV")
    parser.add_argument("--limit", type=int, help="Smoke-only sample limit; omit for formal evaluation")
    parser.add_argument("--out", required=True, type=Path, help="A new, separate output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec = load_experiment(args.config)
    if args.checkpoint and not spec.is_reroute:
        parser.error("Ghost checkpoints apply only to non-monotonic Reroute configs")
    if args.original and args.ablation != "full":
        parser.error("--ablation only applies to checkpoint runs")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    metadata = {"config": spec.config_name, "model": spec.model,
                "routing": spec.routing, "tasks": resolve_tasks(args.tasks),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "ablation": args.ablation, "limit": args.limit,
                "scope": "smoke" if args.limit is not None else "formal"}
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    output = new_output_dir(args.out)
    write_json(output / "run.json", metadata)
    try:
        with mmbench_failure_logging(metadata["tasks"]):
            results = run_evaluation(spec.config_name, tasks=args.tasks, checkpoint=args.checkpoint,
                                     ablation=args.ablation, limit=args.limit)
            if not isinstance(results, dict) or not results.get("results"):
                raise RuntimeError("Evaluation returned no benchmark results")
            write_json(output / "results.json", results)
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
