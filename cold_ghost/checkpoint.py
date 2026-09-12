"""Versioned Ghost-only checkpoints with strict experiment and data identity."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import torch

FORMAT_VERSION = 1
REQUIRED_METADATA = ("backbone", "hidden_size", "num_layers", "drop_layers",
                     "keep_ratios", "ablation", "ghost_config", "manifest_sha256")


def experiment_metadata(spec, modules, manifest_sha256: str) -> dict:
    result = spec.checkpoint_metadata()
    result.update(hidden_size=modules.hidden_size, num_layers=modules.num_layers,
                  ablation=modules.config.ablation, ghost_config=asdict(modules.config),
                  manifest_sha256=manifest_sha256)
    return result


def _validate_metadata(metadata: dict) -> None:
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    digest = metadata["manifest_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("manifest_sha256 must be a lowercase SHA256 digest")
    if metadata["ghost_config"].get("ablation") != metadata["ablation"]:
        raise ValueError("Checkpoint ablation disagrees with ghost_config")


def save_checkpoint(path, modules, metadata: dict, *, optimizer=None,
                    training_state: dict | None = None) -> None:
    """Save only auxiliary weights; never save the frozen backbone."""
    _validate_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "metadata": dict(metadata),
               "state_dict": {key: value.detach().cpu() for key, value in modules.state_dict().items()},
               "training_state": dict(training_state or {})}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect_checkpoint(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Ghost checkpoint format")
    if not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Malformed Ghost checkpoint")
    _validate_metadata(payload["metadata"])
    return payload


def load_checkpoint(path, modules, *, expected_metadata: dict | None = None,
                    optimizer=None, require_complete: bool = True) -> dict:
    payload = inspect_checkpoint(path)
    metadata = payload["metadata"]
    expected = {"hidden_size": modules.hidden_size, "num_layers": modules.num_layers,
                "ghost_config": asdict(modules.config), "ablation": modules.config.ablation}
    expected.update(expected_metadata or {})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch for {key}: expected {value!r}, got {metadata.get(key)!r}")
    state = payload.get("training_state", {})
    if require_complete and (not state.get("complete") or state.get("smoke", False)):
        raise ValueError("Formal evaluation requires a completed two-stage checkpoint, not a smoke/partial checkpoint")
    modules.load_state_dict(payload["state_dict"], strict=True)
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("Resume requires saved optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return payload
