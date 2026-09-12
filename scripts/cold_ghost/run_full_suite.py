#!/usr/bin/env python
"""Run the fixed complete Cold/Ghost training, evaluation and diagnostic suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)

from cold_ghost.full_suite import STAGES, build_plan, child_environment, plan_payload, run_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N; all child processes use that visible GPU selection")
    parser.add_argument("--resume", action="store_true", help="Skip only identity- and artifact-verified completed attempts")
    parser.add_argument("--plan-only", action="store_true", help="List the complete plan without loading data/checkpoints or running jobs")
    args = parser.parse_args(argv)
    for name in ("manifest", "images", "checkpoint_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        child_environment(args.device)
        if args.plan_only:
            payload = plan_payload(build_plan(args.checkpoint_dir), manifest=args.manifest,
                                   images=args.images, checkpoint_dir=args.checkpoint_dir)
            payload["selected_stage"] = args.stage
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        return run_suite(stage=args.stage, manifest=args.manifest, images=args.images,
                         checkpoint_dir=args.checkpoint_dir, out=args.out, device=args.device, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
