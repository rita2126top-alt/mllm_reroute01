"""Exercise a direct CLI process, including real runtime imports."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_direct_doctor_imports_runtime_without_profile_shadowing():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/cold_ghost/doctor.py")],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"]
    assert report["checks"]["cpu_tensor"] == "ok"
