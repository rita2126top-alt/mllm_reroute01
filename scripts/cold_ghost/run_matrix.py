#!/usr/bin/env python3
"""Run 38 original and 24 GC experiment settings, with isolated outputs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)

from cold_ghost.config import ABLATIONS, MODELS, TIERS
from cold_ghost.evaluation import build_matrix, run_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all",) + MODELS, default="all")
    parser.add_argument("--tier", choices=("all",) + TIERS, default="all")
    parser.add_argument("--arms", choices=("all", "original", "gc"), default="all")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--mode", choices=("evaluate", "profile", "benchmark"), default="evaluate")
    parser.add_argument("--n-passes", type=int)
    parser.add_argument("--n-warmup-passes", type=int)
    parser.add_argument("--n-decode-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.arms != "original" and args.checkpoint_dir is None:
        parser.error("GC jobs require --checkpoint-dir (including dry-run plans)")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.mode != "evaluate" and args.limit is not None:
        parser.error("--limit only applies to --mode evaluate")
    if args.mode == "evaluate" and any(value is not None for value in
            (args.n_passes, args.n_warmup_passes, args.n_decode_tokens)):
        parser.error("Repetition/decode flags only apply to efficiency modes")
    if args.n_passes is not None and args.n_passes < 1:
        parser.error("--n-passes must be positive")
    if args.n_warmup_passes is not None and args.n_warmup_passes < 0:
        parser.error("--n-warmup-passes must be non-negative")
    if args.n_decode_tokens is not None and (args.mode != "benchmark" or args.n_decode_tokens < 1):
        parser.error("Positive --n-decode-tokens requires benchmark mode")
    jobs = build_matrix(models=MODELS if args.model == "all" else (args.model,),
        tiers=TIERS if args.tier == "all" else (args.tier,), arms=args.arms,
        checkpoint_dir=args.checkpoint_dir, ablation=args.ablation, tasks=args.tasks)
    return run_matrix(jobs, args.out, limit=args.limit, dry_run=args.dry_run,
        mode=args.mode, n_passes=args.n_passes, n_warmup_passes=args.n_warmup_passes,
        n_decode_tokens=args.n_decode_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
