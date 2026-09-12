"""Export image-content exclusions through original lmms-eval task loaders.

APIs verified against EvolvingLMMs-Lab/lmms-eval tag v0.7.1:
lmms_eval/tasks/__init__.py TaskManager/get_task_dict, and api/task.py
ConfigurableTask.test_docs/validation_docs/doc_to_visual.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import struct

from PIL import Image, ImageOps

from .config import GROUNDING, ROOT
from .data import _write_json, build_exclusions, canonical_sha256

SCOPE_TASKS = {"pope": ("pope",), "gqa": ("gqa",), "mmbench": ("mmbench_en_dev",),
               "mme": ("mme",), "refcoco": GROUNDING[:3],
               "refcoco_plus": GROUNDING[3:6], "refcocog": GROUNDING[6:]}


def pil_content_sha256(image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("doc_to_visual must return PIL images for these original tasks")
    image = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256(struct.pack(">II", *image.size))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _leaves(tree):
    for name, value in tree.items():
        if isinstance(value, dict):
            yield from _leaves(value)
        else:
            yield str(name), value


def export_task_images(task, name, cache_dir, *, every=250, progress=print):
    """Resume at a document boundary, verified against dataset fingerprint."""
    if task.has_test_docs():
        docs, split = task.test_docs(), task.config.test_split
    elif task.has_validation_docs():
        docs, split = task.validation_docs(), task.config.validation_split
    else:
        raise ValueError(f"Evaluation task has no test/validation split: {name}")
    if len(docs) < 1:
        raise ValueError(f"Empty evaluation task: {name}")
    identity = {"task": name, "split": split, "doc_count": len(docs),
                "dataset_fingerprint": getattr(docs, "_fingerprint", None),
                "dataset_path": task.config.dataset_path, "dataset_name": task.config.dataset_name,
                "lmms_eval_version": "0.7.1"}
    target = Path(cache_dir) / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".json")
    state = {"identity": identity, "cursor": 0, "complete": False, "content_sha256": []}
    if target.exists():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("identity") != identity:
            raise ValueError(f"Cached dataset identity changed: {name}. Use a fresh cache directory.")
        checksum = cached.pop("cache_sha256", None)
        if checksum != canonical_sha256(cached):
            raise ValueError(f"Corrupt exclusion resume cache: {target}")
        state = cached
    hashes = set(state["content_sha256"])

    def save():
        state["content_sha256"] = sorted(hashes)
        _write_json(target, state | {"cache_sha256": canonical_sha256(state)})

    for index in range(state["cursor"], len(docs)):
        visuals = task.doc_to_visual(docs[index])
        if isinstance(visuals, Image.Image):
            visuals = [visuals]
        if not visuals:
            raise ValueError(f"No evaluation image for {name} document {index}")
        for visual in visuals:
            hashes.add(pil_content_sha256(visual))
        state["cursor"] = index + 1
        if state["cursor"] % every == 0:
            save()
            progress(f"{name}: {state['cursor']}/{len(docs)} documents, {len(hashes)} image hashes")
    state["complete"] = True
    save()
    progress(f"{name}: complete, {len(docs)} documents, {len(hashes)} distinct images")
    return state


def auto_exclusions(output, cache_dir, *, task_manager=None, task_loader=None, progress=print):
    if task_manager is None:
        installed = version("lmms_eval")
        if installed != "0.7.1":
            raise RuntimeError(f"Expected lmms-eval 0.7.1, found {installed}")
        from lmms_eval.tasks import TaskManager, get_task_dict
        task_manager, task_loader = TaskManager(model_name="llava_hf"), get_task_dict
    if task_loader is None:
        raise ValueError("A supplied task manager requires a task loader")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {"format_version": 1, "smoke": False, "scopes": {},
               "source": "lmms-eval v0.7.1 original evaluation task doc_to_visual"}
    # Load one family at a time, retaining cached hashes rather than every dataset.
    for scope, task_names in SCOPE_TASKS.items():
        loaded = task_loader(list(task_names), task_manager=task_manager)
        hashes, sources = set(), []
        for name, task in _leaves(loaded):
            state = export_task_images(task, name, cache_dir, progress=progress)
            hashes.update(state["content_sha256"])
            sources.append(state["identity"])
        if not hashes:
            raise ValueError(f"No images exported for scope {scope}")
        result["scopes"][scope] = {"sources": sources, "image_count": len(hashes), "content_sha256": sorted(hashes)}
        del loaded
    efficiency = build_exclusions({"efficiency": [str(ROOT / "bench_data" / "manifest.json")]},
                                  cache_dir / "efficiency.json", smoke=True)
    result["scopes"]["efficiency"] = efficiency["scopes"]["efficiency"]
    result["sha256"] = canonical_sha256(result)
    _write_json(output, result)
    return result
