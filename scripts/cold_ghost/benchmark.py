#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)
from cold_ghost.profiling import cli

if __name__ == "__main__":
    raise SystemExit(cli("benchmark"))
