#!/usr/bin/env python
"""Automatically hash images from all release evaluation tasks and cohort."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/cold_ghost/exclusions.json")
    parser.add_argument("--cache-dir", default="data/cold_ghost/exclusion_cache")
    args = parser.parse_args()
    from cold_ghost.exclusions import auto_exclusions
    result = auto_exclusions(args.out, args.cache_dir)
    print(json.dumps({"output": args.out, "sha256": result["sha256"],
                      "scopes": {name: data["image_count"] for name, data in result["scopes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
