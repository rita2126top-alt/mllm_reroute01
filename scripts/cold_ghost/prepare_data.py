#!/usr/bin/env python
"""Prepare auditable GQA training data; never train on benchmark splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)

from cold_ghost.data import build_exclusions, prepare_gqa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    exclude = subparsers.add_parser("exclusions", help="Hash benchmark image content before selecting training images")
    exclude.add_argument("--scope", action="append", required=True, metavar="NAME=PATH",
                         help="Repeated scope=directory or scope=image-path-manifest; all 8 scopes required")
    exclude.add_argument("--out", required=True)
    exclude.add_argument("--smoke", action="store_true")
    prepare = subparsers.add_parser("gqa", help="Choose deterministic train8192/validation512 image/question prompts")
    prepare.add_argument("--questions", required=True, help="Official train_balanced_questions.json")
    prepare.add_argument("--images", required=True, help="Directory containing GQA imageID.jpg files")
    prepare.add_argument("--exclusions", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--train-count", type=int, default=8192)
    prepare.add_argument("--val-count", type=int, default=512)
    prepare.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "exclusions":
        scopes = {}
        for item in args.scope:
            if "=" not in item:
                parser.error("--scope must be NAME=PATH")
            name, path = item.split("=", 1)
            scopes.setdefault(name, []).append(path)
        result = build_exclusions(scopes, args.out, smoke=args.smoke)
        print(json.dumps({"output": args.out, "sha256": result["sha256"],
                          "scopes": {name: item["image_count"] for name, item in result["scopes"].items()}}, indent=2))
    else:
        result = prepare_gqa(args.questions, args.images, args.exclusions, args.out,
                             train_count=args.train_count, val_count=args.val_count, smoke=args.smoke)
        print(json.dumps({"output": args.out, "manifest_sha256": result["manifest_sha256"],
                          "train": len(result["train"]), "validation": len(result["validation"]),
                          "smoke": result["smoke"]}, indent=2))


if __name__ == "__main__":
    main()
