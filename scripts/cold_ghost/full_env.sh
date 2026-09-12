#!/usr/bin/env bash
# Source this file after activating the dedicated Conda environment.
# Every path can be overridden before sourcing, except the actual project root.
export GC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GC_DATA_ROOT="${GC_DATA_ROOT:-$HOME/mllm_data}"
export HF_HOME="${HF_HOME:-$GC_DATA_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export GQA_ROOT="${GQA_ROOT:-$GC_DATA_ROOT/gqa}"
export GQA_IMAGES="${GQA_IMAGES:-$GQA_ROOT/images}"
export GC_MANIFEST="${GC_MANIFEST:-$GC_PROJECT_ROOT/data/cold_ghost/gqa_8192_512.json}"
export GC_EXCLUSIONS="${GC_EXCLUSIONS:-$GC_PROJECT_ROOT/data/cold_ghost/exclusions.json}"
export GC_EXCLUSION_CACHE="${GC_EXCLUSION_CACHE:-$GC_PROJECT_ROOT/data/cold_ghost/exclusion_cache}"
export GC_CHECKPOINT_DIR="${GC_CHECKPOINT_DIR:-$GC_PROJECT_ROOT/checkpoints/cold_ghost}"
export GC_RUN_ROOT="${GC_RUN_ROOT:-$GC_PROJECT_ROOT/experiments/cold_ghost/full_run}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export CONDA_ENV="${CONDA_DEFAULT_ENV:-mllm_reroute_gc}"
export PYTHONUNBUFFERED=1

# Original MMBench judge protocol; the secret API key is supplied interactively.
export API_TYPE="${API_TYPE:-openai}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}"
export MODEL_VERSION="${MODEL_VERSION:-gpt-4o-2024-11-20}"
