"""Deterministic, answer-free GQA training manifests and image exclusion audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from PIL import Image, ImageOps

EVALUATION_SCOPES = ("pope", "gqa", "mmbench", "mme", "refcoco", "refcoco_plus", "refcocog", "efficiency")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_content_sha256(path) -> str:
    """Hash decoded, orientation-corrected RGB pixels, not file metadata."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(struct.pack(">II", *image.size))
        digest.update(image.tobytes())
        return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _image_paths(source) -> list[Path]:
    """Read an image directory or a UTF-8 text/JSON/JSONL image-path manifest.

    JSON records use ``image`` or ``image_path``. Relative paths are relative
    to the manifest, allowing the release bench_data/manifest.json verbatim.
    A JSON GQA annotation mapping is also supported when each value contains
    an explicit image_path; ordinary annotations need a separate path list.
    """
    source = Path(source).resolve()
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        paths = [source]
    else:
        content = source.read_text(encoding="utf-8-sig")
        if source.suffix.lower() == ".json":
            records = json.loads(content)
            if isinstance(records, dict):
                records = list(records.values())
        elif source.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            records = [line.strip() for line in content.splitlines() if line.strip()]
        paths = []
        for record in records:
            raw = record if isinstance(record, str) else record.get("image_path", record.get("image"))
            if not isinstance(raw, str):
                raise ValueError(f"An exclusion record needs an image path: {source}")
            path = Path(raw)
            paths.append(path if path.is_absolute() else source.parent / path)
    if not paths:
        raise ValueError(f"Empty evaluation image source: {source}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def build_exclusions(scopes: dict[str, list[str]], output, *, smoke=False) -> dict:
    missing = set(EVALUATION_SCOPES) - set(scopes)
    if missing and not smoke:
        raise ValueError(f"Evaluation image exclusion sources missing: {sorted(missing)}")
    result = {"format_version": 1, "smoke": bool(smoke), "scopes": {}}
    cache = {}
    for name, sources in sorted(scopes.items()):
        hashes = set()
        for source in sources:
            for path in _image_paths(source):
                key = str(path.resolve())
                if key not in cache:
                    cache[key] = image_content_sha256(path)
                hashes.add(cache[key])
        result["scopes"][name] = {"sources": sources, "image_count": len(hashes),
                                   "content_sha256": sorted(hashes)}
    result["sha256"] = canonical_sha256(result)
    _write_json(output, result)
    return result


def _load_exclusions(path, smoke: bool) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = value.pop("sha256", None)
    if digest != canonical_sha256(value):
        raise ValueError("Exclusion manifest SHA256 does not match its contents")
    if not smoke and (value.get("smoke") or set(EVALUATION_SCOPES) - set(value.get("scopes", {}))):
        raise ValueError("Formal data preparation requires all evaluation scopes and a non-smoke exclusion audit")
    for name, scope in value.get("scopes", {}).items():
        hashes = scope.get("content_sha256", [])
        if not hashes or scope.get("image_count") != len(hashes):
            raise ValueError(f"Invalid/empty exclusion scope {name}")
    value["sha256"] = digest
    return value


def prepare_gqa(questions_path, image_root, exclusions_path, output, *,
                train_count=8192, val_count=512, smoke=False) -> dict:
    """Select one minimum numeric question ID per image, then deduplicate.

    Formal runs require the official ``train_balanced_questions.json``
    source and the fixed 8192/512 split; smoke manifests cannot be evaluated
    as trained experiment checkpoints.
    """
    questions_path, image_root = Path(questions_path), Path(image_root).resolve()
    if train_count < 1 or val_count < 1:
        raise ValueError("Training and independent validation must each contain at least one image")
    if not smoke and (train_count, val_count) != (8192, 512):
        raise ValueError("Formal manifests require exactly 8192 train / 512 validation images")
    if "train_balanced" not in questions_path.name:
        raise ValueError("Only GQA train_balanced question annotations are accepted")
    exclusions = _load_exclusions(exclusions_path, smoke)
    forbidden = {digest for scope in exclusions["scopes"].values() for digest in scope["content_sha256"]}
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, dict):
        raise ValueError("GQA questions must be a mapping from numeric question ID to record")
    chosen = {}
    for question_id, record in questions.items():
        if not str(question_id).isdigit():
            raise ValueError(f"Non-numeric GQA question ID: {question_id}")
        image_id = str(record["imageId"])
        if image_id not in chosen or int(question_id) < int(chosen[image_id][0]):
            chosen[image_id] = (str(question_id), record["question"])
    # Hash ordering also defines the deterministic representative of content duplicates.
    ordered_ids = sorted(chosen, key=lambda identifier: (hashlib.sha256(("42:" + identifier).encode()).hexdigest(), identifier))
    records, seen, excluded_count, duplicate_count = [], set(), 0, 0
    for image_id in ordered_ids:
        path = image_root / f"{image_id}.jpg"
        if not path.is_file():
            matches = [image_root / f"{image_id}{suffix}" for suffix in sorted(IMAGE_EXTENSIONS)]
            path = next((candidate for candidate in matches if candidate.is_file()), path)
        if not path.is_file():
            raise FileNotFoundError(f"Training image is missing: {path}")
        content_digest = image_content_sha256(path)
        if content_digest in forbidden:
            excluded_count += 1
            continue
        if content_digest in seen:
            duplicate_count += 1
            continue
        seen.add(content_digest)
        question_id, question = chosen[image_id]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question text for {question_id}")
        records.append({"image_id": image_id, "question_id": question_id, "question": question,
                        "image": path.relative_to(image_root).as_posix(),
                        "content_sha256": content_digest, "file_sha256": file_sha256(path)})
        if len(records) == train_count + val_count:
            break
    if len(records) < train_count + val_count:
        raise ValueError(f"Only {len(records)} eligible distinct images; need {train_count + val_count}. Never supplement from evaluation splits.")
    payload = {"format_version": 1, "source_split": "gqa/train_balanced", "seed": 42,
               "smoke": bool(smoke), "question_file_sha256": file_sha256(questions_path),
               "exclusions_sha256": exclusions["sha256"], "exclusion_scopes": list(exclusions["scopes"]),
               "selection": "min_numeric_question_per_image;sha256(42:image_id);deduplicate_rgb_content",
               "excluded_before_cutoff": excluded_count, "duplicates_before_cutoff": duplicate_count,
               "train": records[:train_count], "validation": records[train_count:]}
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(output, payload)
    return payload


def load_manifest(path, *, allow_smoke=False) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = payload.pop("manifest_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("Training manifest SHA256 does not match its contents")
    if payload.get("source_split") != "gqa/train_balanced" or payload.get("seed") != 42:
        raise ValueError("Unsupported training manifest source/seed")
    if payload.get("smoke") and not allow_smoke:
        raise ValueError("Smoke manifest requires explicit --smoke")
    train, validation = payload.get("train", []), payload.get("validation", [])
    if not train or not validation:
        raise ValueError("Manifest must contain independent training and validation sets")
    if not payload.get("smoke") and (len(train), len(validation)) != (8192, 512):
        raise ValueError("Incorrect formal split sizes")
    rows = train + validation
    if len({row["content_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image content in/across train and validation")
    for row in rows:
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest image paths must stay within image_root")
        if "answer" in row or "answers" in row:
            raise ValueError("Training records must contain prompts only")
    payload["manifest_sha256"] = digest
    return payload


class PromptDataset:
    """Return PIL image plus question, verifying image bytes on first access."""

    def __init__(self, manifest, image_root, split="train", *, verify_images=True):
        if split not in ("train", "validation"):
            raise ValueError("Training tools only read train or internal validation")
        self.rows = manifest[split]
        self.root = Path(image_root).resolve()
        self.verify_images = verify_images
        self._verified = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["image"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path leaves image root")
        if self.verify_images and index not in self._verified:
            if file_sha256(path) != row["file_sha256"] or image_content_sha256(path) != row["content_sha256"]:
                raise ValueError(f"Image changed since data preparation: {path}")
            self._verified.add(index)
        with Image.open(path) as image:
            result = ImageOps.exif_transpose(image).convert("RGB")
        return {"image": result, "question": row["question"], "image_id": row["image_id"]}
