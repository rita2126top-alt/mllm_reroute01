#!/usr/bin/env bash
# Run inside the activated mllm_reroute_gc Conda environment on Linux.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(uname -s)" != Linux ]]; then
    echo "This installer targets the Linux GPU server. See requirements-cpu.txt for local tests." >&2
    exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == base ]]; then
    echo "Activate the dedicated mllm_reroute_gc Conda environment first." >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the release server environment"'
python -m pip install -c environments/cold_ghost/constraints.txt \
    -r requirements.txt -r environments/cold_ghost/requirements-extra.txt
LMMS_SOURCE="${LMMS_SOURCE:-$(dirname "$PROJECT_ROOT")/lmms-eval-v0.7.1}"
if [[ ! -d "$LMMS_SOURCE/.git" ]]; then
    if [[ -e "$LMMS_SOURCE" ]]; then
        echo "Existing non-Git path: $LMMS_SOURCE. Set LMMS_SOURCE to a fresh sibling directory." >&2
        exit 1
    fi
    git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$LMMS_SOURCE"
fi
LMMS_TAG="$(git -C "$LMMS_SOURCE" describe --tags --exact-match HEAD)"
if [[ "$LMMS_TAG" != v0.7.1 ]]; then
    echo "Expected lmms-eval v0.7.1, found $LMMS_TAG in $LMMS_SOURCE." >&2
    exit 1
fi
python -m pip install -c environments/cold_ghost/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 -e "$LMMS_SOURCE"
bash lmms_eval_patches/apply.sh "$LMMS_SOURCE/lmms_eval"
python -m pip check
python scripts/cold_ghost/doctor.py --require-gpu --require-eval
python scripts/cold_ghost/check_project.py
