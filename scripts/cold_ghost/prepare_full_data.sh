#!/usr/bin/env bash
# Prepare the complete, fixed Cold/Ghost training data on the Linux server.
set -euo pipefail

# Load the shared project/data/cache paths used by every formal-run wrapper.
GC_WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GC_WRAPPER_DIR/full_env.sh"
cd "$GC_PROJECT_ROOT"

# Fail before downloads if the activated environment lacks required programs.
for GC_REQUIRED_PROGRAM in python hf wget unzip find mktemp; do
    command -v "$GC_REQUIRED_PROGRAM" >/dev/null 2>&1 || {
        echo "Missing program: $GC_REQUIRED_PROGRAM. Activate the installed mllm_reroute_gc Conda environment first." >&2
        exit 1
    }
done

# Keep all archives, model caches and final manifests at the shared locations.
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$GQA_ROOT" \
    "$(dirname "$GC_MANIFEST")" "$(dirname "$GC_EXCLUSIONS")" "$GC_EXCLUSION_CACHE"

# Validate an explicit question-file override before starting large downloads.
if [[ -n "${GQA_QUESTIONS:-}" ]]; then
    if [[ ! -f "$GQA_QUESTIONS" || ! -r "$GQA_QUESTIONS" || "$(basename "$GQA_QUESTIONS")" != train_balanced_questions.json ]]; then
        echo "GQA_QUESTIONS must name a readable official train_balanced_questions.json file: $GQA_QUESTIONS" >&2
        exit 1
    fi
fi

# Download both ORIGINAL pretrained backbones into the shared Hugging Face cache.
# Repeating these commands reuses files already present in that cache.
hf download llava-hf/llava-1.5-7b-hf
hf download Qwen/Qwen2.5-VL-7B-Instruct

# Download the official raw GQA images and question annotations with resumption.
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -O "$GQA_ROOT/questions1.2.zip"
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O "$GQA_ROOT/images.zip"

# Extract missing files; -n prevents overwriting previously extracted data.
unzip -n "$GQA_ROOT/questions1.2.zip" -d "$GQA_ROOT"
unzip -n "$GQA_ROOT/images.zip" -d "$GQA_ROOT"

# Clean incomplete temporary files; a differing valid candidate is kept for audit.
GC_FIND_RESULTS=""
GC_MANIFEST_CANDIDATE=""
GC_KEEP_CANDIDATE=0
gc_cleanup_data_temporaries() {
    if [[ -n "$GC_FIND_RESULTS" && -f "$GC_FIND_RESULTS" ]]; then
        rm -- "$GC_FIND_RESULTS"
    fi
    if [[ "$GC_KEEP_CANDIDATE" == 0 && -n "$GC_MANIFEST_CANDIDATE" && -f "$GC_MANIFEST_CANDIDATE" ]]; then
        rm -- "$GC_MANIFEST_CANDIDATE"
    fi
}
trap gc_cleanup_data_temporaries EXIT

# Without an override, require exactly one official balanced training file.
# A NUL-delimited temporary listing preserves paths containing spaces/newlines,
# and the direct find command makes lookup errors fail under set -e.
if [[ -z "${GQA_QUESTIONS:-}" ]]; then
    GC_FIND_RESULTS="$(mktemp "${TMPDIR:-/tmp}/gc-gqa-questions.XXXXXX")"
    find "$GQA_ROOT" -type f -name train_balanced_questions.json -print0 > "$GC_FIND_RESULTS"
    mapfile -d '' -t GC_QUESTION_FILES < "$GC_FIND_RESULTS"
    if [[ "${#GC_QUESTION_FILES[@]}" -ne 1 ]]; then
        echo "Expected exactly one train_balanced_questions.json under $GQA_ROOT; found ${#GC_QUESTION_FILES[@]}." >&2
        echo "Set GQA_QUESTIONS to the intended official training file if multiple copies exist; evaluation question files are not accepted." >&2
        exit 1
    fi
    export GQA_QUESTIONS="${GC_QUESTION_FILES[0]}"
fi

# Check that GQA_IMAGES directly contains readable image files, as required by
# imageID-based manifest preparation; every selected image is checked again later.
python - "$GQA_QUESTIONS" "$GQA_IMAGES" <<'PY'
from pathlib import Path
import sys

questions, images = map(Path, sys.argv[1:])
if questions.name != "train_balanced_questions.json" or not questions.is_file():
    raise SystemExit(f"Not the official balanced training question-file path: {questions}")
if not images.is_dir():
    raise SystemExit(f"GQA_IMAGES directory does not exist: {images}")
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
if not any(path.is_file() and path.suffix.lower() in extensions for path in images.iterdir()):
    raise SystemExit(f"GQA_IMAGES must directly contain imageID.jpg files, not another enclosing directory: {images}")
print(f"GQA training questions: {questions}")
print(f"GQA image directory: {images}")
PY

# Export COMPLETE evaluation-image exclusions through original lmms-eval tasks.
# This includes all 12 task entries and the bundled efficiency images; the
# exporter resumes compatible dataset-fingerprint/document-cursor caches.
python scripts/cold_ghost/export_exclusions.py \
    --out "$GC_EXCLUSIONS" --cache-dir "$GC_EXCLUSION_CACHE"

# Prepare the fixed 8192/512 split into a same-directory temporary file first.
# The target's existing contents and timestamp remain untouched during preparation.
GC_MANIFEST_CANDIDATE="$(mktemp "${GC_MANIFEST}.candidate.XXXXXX")"
python scripts/cold_ghost/prepare_data.py gqa \
    --questions "$GQA_QUESTIONS" --images "$GQA_IMAGES" \
    --exclusions "$GC_EXCLUSIONS" --out "$GC_MANIFEST_CANDIDATE"

# Validate both full manifests. Publish a first manifest atomically, retain an
# identical existing manifest, or stop and keep a different candidate for audit.
# A different manifest requires a new GC_MANIFEST path and matching new training
# outputs; it must never silently change the data identity of existing checkpoints.
GC_KEEP_CANDIDATE=1
python - "$GC_MANIFEST" "$GC_MANIFEST_CANDIDATE" <<'PY'
import os
from pathlib import Path
import sys

from cold_ghost.data import load_manifest

target, candidate = map(Path, sys.argv[1:])
prepared = load_manifest(candidate, allow_smoke=False)
if target.exists():
    existing = load_manifest(target, allow_smoke=False)
    if existing["manifest_sha256"] != prepared["manifest_sha256"]:
        raise SystemExit(
            "Formal training manifest differs from the existing data identity. "
            f"Original kept: {target}\nCandidate kept: {candidate}\n"
            "Set GC_MANIFEST to a NEW path and use separate compatible training/output paths; "
            "do not overwrite the manifest associated with existing checkpoints."
        )
    candidate.unlink()
    print(f"Existing identical formal manifest retained: {target}")
else:
    os.replace(candidate, target)
    print(f"Created formal training manifest: {target}")
print(f"manifest_sha256={prepared['manifest_sha256']}")
print(f"train={len(prepared['train'])}, validation={len(prepared['validation'])}")
PY

# The formal preparation is complete; subsequent full training reads GC_MANIFEST.
echo "Full data preparation completed: $GC_MANIFEST"
