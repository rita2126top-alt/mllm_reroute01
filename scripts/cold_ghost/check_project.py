#!/usr/bin/env python
"""Verify original-file integrity, Python/Bash syntax, CLI imports and CPU tests."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)


def original_integrity(root=ROOT):
    manifest = json.loads((root / "environments/cold_ghost/original_files_sha256.json").read_text(encoding="utf-8"))
    differences = []
    for name, expected in manifest["files"].items():
        target = root / name
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        if actual != expected:
            differences.append(name)
    return {"baseline_commit": manifest["baseline_commit"], "checked": len(manifest["files"]),
            "changed_or_missing": differences, "ok": not differences}


def command_record(command, *, env=None, input_text=None):
    result = subprocess.run(command, cwd=ROOT, env=env, input=input_text,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "ok": result.returncode == 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax-only", action="store_true",
                        help="Skip imports/tests; this alone does not establish runtime correctness")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/cold_ghost/checks/report.json")
    args = parser.parse_args(argv)
    report = {"original_integrity": original_integrity(), "python": [], "bash": [], "cli": []}
    roots = ("models", "scripts", "profiler", "cold_ghost", "tests/cold_ghost")
    for directory in roots:
        for path in sorted((ROOT / directory).rglob("*.py")):
            record = {"path": path.relative_to(ROOT).as_posix()}
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                record["ok"] = True
            except (OSError, SyntaxError, UnicodeError) as exc:
                record.update(ok=False, error=str(exc))
            report["python"].append(record)
    bash = shutil.which("bash")
    if not bash and os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parent.parent / "bin/bash.exe" if git else Path("")
        if candidate.is_file():
            bash = str(candidate)
    if bash:
        for folder in ("scripts", "lmms_eval_patches"):
            for path in sorted((ROOT / folder).rglob("*.sh")):
                record = command_record([bash, "-n"], input_text=path.read_text(encoding="utf-8"))
                record["path"] = path.relative_to(ROOT).as_posix()
                report["bash"].append(record)
    else:
        report["bash_unavailable"] = True
    if not args.syntax_only:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONUTF8="1")
        for path in sorted((ROOT / "scripts/cold_ghost").glob("*.py")):
            report["cli"].append(command_record([sys.executable, str(path), "--help"], env=env))
        report["runtime"] = command_record(
            [sys.executable, str(ROOT / "scripts/cold_ghost/doctor.py")], env=env)
        report["cpu_tests"] = command_record(
            [sys.executable, "-m", "pytest", "-q", "tests/cold_ghost"], env=env)
    checks = [report["original_integrity"]] + report["python"] + report["bash"] + report["cli"]
    if "cpu_tests" in report:
        checks.extend([report["runtime"], report["cpu_tests"]])
    report["ok"] = all(check["ok"] for check in checks)
    report["syntax_only"] = args.syntax_only
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "original_files": report["original_integrity"]["checked"],
                      "python_files": len(report["python"]), "bash_files": len(report["bash"]),
                      "cli_entries": len(report["cli"]), "report": str(args.report)}, indent=2))
    if not report["ok"]:
        for record in checks:
            if not record["ok"]:
                print(json.dumps(record, ensure_ascii=False, indent=2), file=sys.stderr)
    elif "cpu_tests" in report:
        print(report["cpu_tests"]["stdout"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
