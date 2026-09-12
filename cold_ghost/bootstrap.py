"""Keep direct script entry points from shadowing standard-library modules."""
from pathlib import Path
import sys


def isolate_script_directory(script_file):
    directory = Path(script_file).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() != directory
    ]
