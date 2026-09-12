#!/usr/bin/env bash
# Full official experiment protocol. No shortened training or sample limits.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/cold_ghost/run_full.sh [STAGE] [--resume]
STAGE: all (default), prepare, train, accuracy, profile, benchmark, ablation, diagnostics

all: prepare complete data, train 60 checkpoints, then run all 246 experiment jobs.
prepare: download both backbones and GQA, export full exclusions, build 8192/512 manifest.
train: 2 backbones x 2 schedules x 3 budgets x 5 variants = 60 checkpoints.
accuracy: 38 original + 24 full GC settings, all 12 task entries, no sample limit.
profile: same 62 settings, original 3-image cohort, 1 pass, 0 warmup passes.
benchmark: same 62 settings, original cohort, 5 passes, 2 warmup, 64 generated tokens.
ablation: 48 non-full compact settings, full original four-task ablation protocol.
diagnostics: all 12 compact settings, all 512 independent validation images.

--resume: validate and reuse completed jobs; retry failed jobs in new attempt directories.
Activate the Conda environment first. Paths/device are set in full_env.sh or exported variables.
HELP
}

GC_STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$GC_STAGE" in
    -h|--help) usage; exit 0 ;;
    all|prepare|train|accuracy|profile|benchmark|ablation|diagnostics) ;;
    *) usage >&2; exit 2 ;;
esac
GC_RESUME_ARGS=()
if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && "$1" == --resume ]]; then
        GC_RESUME_ARGS=(--resume)
    else
        usage >&2
        exit 2
    fi
fi

# Resolve all shared paths and select one GPU.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/full_env.sh"
cd "$GC_PROJECT_ROOT"
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == base ]]; then
    echo "Activate the dedicated mllm_reroute_gc Conda environment first." >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "The full server protocol requires Python 3.10"'

# Preparation requires the evaluation task loaders, but does not execute a model.
if [[ "$GC_STAGE" == prepare ]]; then
    python scripts/cold_ghost/doctor.py --require-eval
    bash scripts/cold_ghost/prepare_full_data.sh
    exit 0
fi

# Check judge configuration before a long run, without making a billable request.
if [[ "$GC_STAGE" == all || "$GC_STAGE" == accuracy || "$GC_STAGE" == ablation ]]; then
    python -c "from cold_ghost.judge import validate_mmbench_judge_environment; validate_mmbench_judge_environment(); print('MMBench judge configuration checked; no API request sent.')"
fi

# All formal model stages require the fixed GPU runtime and real task factories.
python scripts/cold_ghost/doctor.py --require-gpu --require-eval
if [[ "$GC_STAGE" == all ]]; then
    bash scripts/cold_ghost/prepare_full_data.sh
fi

# The Python runner owns logs, per-attempt outputs, result validation and resume.
python scripts/cold_ghost/run_full_suite.py \
    --stage "$GC_STAGE" \
    --manifest "$GC_MANIFEST" \
    --images "$GQA_IMAGES" \
    --checkpoint-dir "$GC_CHECKPOINT_DIR" \
    --out "$GC_RUN_ROOT" \
    --device cuda \
    "${GC_RESUME_ARGS[@]}"
