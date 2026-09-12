#!/usr/bin/env python
"""Check the installed runtime without downloading models or datasets."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)


def inspect_runtime(require_gpu=False, require_eval=False):
    report = {"python": sys.version, "platform": platform.platform(), "checks": {}, "errors": []}
    packages = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.4.0",
                "hydra-core": "1.3.2", "omegaconf": "2.3.0", "PyYAML": "6.0.3"}
    if require_eval:
        packages.update({"lmms-eval": "0.7.1", "antlr4-python3-runtime": "4.9.3",
                         "latex2sympy2": "1.5.4", "math-verify": "0.9.0",
                         "latex2sympy2-extended": "1.11.0"})
    for package, expected in packages.items():
        try:
            version = importlib.metadata.version(package)
            report["checks"][package] = version
            if version.split("+")[0] != expected:
                report["errors"].append(f"{package}: expected {expected}, installed {version}")
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {package}")
    for name in ("torch", "torchvision", "transformers", "yaml", "cold_ghost.ghost",
                 "cold_ghost.patching", "cold_ghost.training", "cold_ghost.evaluation"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            report["errors"].append(f"Import {name}: {type(exc).__name__}: {exc}")
    try:
        import torch
        reference = torch.ones(2, 2)
        assert torch.equal(reference @ reference, torch.full((2, 2), 2.0))
        report["checks"]["cpu_tensor"] = "ok"
        report["checks"]["torch_runtime_version"] = torch.__version__
        if torch.__version__ != importlib.metadata.version("torch"):
            report["errors"].append("torch runtime files and installed distribution metadata disagree")
        report["checks"]["cuda_available"] = torch.cuda.is_available()
        report["checks"]["cuda_runtime"] = torch.version.cuda
        if require_gpu:
            if not torch.cuda.is_available():
                report["errors"].append("CUDA GPU unavailable; check the driver and the cu128 wheel")
            else:
                device = torch.ones(2, 2, device="cuda")
                result = device @ device
                torch.cuda.synchronize()
                assert float(result[0, 0]) == 2.0
                report["checks"]["gpu"] = torch.cuda.get_device_name(0)
                report["checks"]["gpu_bytes"] = torch.cuda.get_device_properties(0).total_memory
                report["checks"]["bf16_supported"] = torch.cuda.is_bf16_supported()
                if torch.version.cuda != "12.8":
                    report["errors"].append("The fixed server environment requires the CUDA 12.8 PyTorch wheel")
    except Exception as exc:
        report["errors"].append(f"Tensor runtime: {type(exc).__name__}: {exc}")
    if require_eval:
        try:
            import lmms_eval
            from lmms_eval.tasks import TaskManager
            from lmms_eval import evaluator
            from lmms_eval.models import get_model
            from cold_ghost.config import resolve_tasks
            if not callable(evaluator.simple_evaluate):
                raise TypeError("lmms-eval simple_evaluate API is unavailable")
            for model_name in ("llava_hf", "qwen2_5_vl"):
                model_class = get_model(model_name)
                if not callable(getattr(model_class, "create_from_arg_string", None)):
                    raise TypeError(f"{model_name} lacks the release model-loader API")
            report["checks"]["evaluation_model_factories"] = "imported without creating models"
            manager = TaskManager()
            known = set(manager.all_tasks)
            missing = set(resolve_tasks("all")) - known
            if missing:
                report["errors"].append(f"Missing evaluation task definitions: {sorted(missing)}")
            location = Path(lmms_eval.__file__).resolve().parent
            for task in ("refcoco", "refcoco+", "refcocog"):
                target = location / "tasks" / task / "utils_rec.py"
                source = target.read_text(encoding="utf-8")
                if "BBOX_COORD_FORMAT" not in source or "REFCOCO_PROMPT_STYLE" not in source:
                    report["errors"].append(f"Original grounding patch missing: {task}")
            report["checks"]["evaluation_task_registry"] = "checked without loading datasets"
        except Exception as exc:
            report["errors"].append(f"Evaluation harness: {type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = inspect_runtime(args.require_gpu, args.require_eval)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
