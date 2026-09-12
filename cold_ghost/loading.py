"""Model and checkpoint loading shared by the new GC entry points."""
from __future__ import annotations

from pathlib import Path
from .config import ExperimentSpec


def make_router(spec: ExperimentSpec):
    from scripts.run_eval import build_router
    return build_router(spec.as_omegaconf())


def validate_checkpoint(spec: ExperimentSpec, path: str | Path, ablation="full") -> dict:
    """Fail before loading 7B weights if checkpoint provenance is incompatible."""
    if not spec.is_reroute:
        raise ValueError("Cold/Ghost requires an original non-monotonic Reroute config")
    from .checkpoint import inspect_checkpoint
    payload = inspect_checkpoint(path)
    meta = payload.get("metadata", {})
    expected = {"backbone": spec.model["pretrained"],
                "num_layers": spec.model["num_layers"],
                "drop_layers": list(spec.routing["drop_layers"]),
                "keep_ratios": list(spec.routing["keep_ratios"]), "ablation": ablation}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch: expected {value!r}, got {meta.get(key)!r}")
    state = payload.get("training_state", {})
    if not state.get("complete") or state.get("smoke", False):
        raise ValueError("Formal inference requires a complete, non-smoke Ghost checkpoint")
    if not meta.get("manifest_sha256") or not isinstance(meta.get("ghost_config"), dict):
        raise ValueError("Checkpoint lacks training manifest or Ghost configuration provenance")
    from dataclasses import asdict
    from .ghost import GhostConfig
    if meta["ghost_config"] != asdict(GhostConfig(ablation=ablation)):
        raise ValueError("Formal evaluation requires the fixed published Ghost hyperparameters")
    return payload


def _text_config(model):
    cfg = model.config
    for attr in ("text_config", "language_config"):
        nested = getattr(cfg, attr, None)
        if nested is not None:
            return nested
    return cfg


def patch_experiment(model, spec: ExperimentSpec, checkpoint=None, ablation="full"):
    router = make_router(spec)
    if checkpoint is None:
        if router is None:
            return None
        from models.patching import patch_model_for_routing
        from models.dispatcher import TokenDispatcher
        return patch_model_for_routing(model, router, TokenDispatcher(),
            action=spec.routing["action"], model_family=spec.model_family)
    payload = validate_checkpoint(spec, checkpoint, ablation)
    from .checkpoint import load_checkpoint
    from .ghost import GhostConfig, GhostModules
    from .patching import patch_model_with_cold_ghost
    cfg = _text_config(model)
    hidden_size = int(cfg.hidden_size)
    num_layers = int(cfg.num_hidden_layers)
    modules = GhostModules(hidden_size, num_layers,
                           GhostConfig(**payload["metadata"]["ghost_config"]))
    modules.to(device=next(model.parameters()).device).eval()
    load_checkpoint(checkpoint, modules, expected_metadata={
        "backbone": spec.model["pretrained"], "hidden_size": hidden_size,
        "num_layers": num_layers, "drop_layers": list(spec.routing["drop_layers"]),
        "keep_ratios": list(spec.routing["keep_ratios"]), "ablation": ablation},
        require_complete=True)
    return patch_model_with_cold_ghost(model, router, action=spec.routing["action"],
        model_family=spec.model_family, ghost_modules=modules)


def load_backbone(spec: ExperimentSpec, device="cuda"):
    """Use the same processors, SDPA and backbone dtypes as release profilers."""
    import torch
    from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                              Qwen2_5_VLForConditionalGeneration)
    is_llava = spec.model_family == "llava"
    cls = LlavaForConditionalGeneration if is_llava else Qwen2_5_VLForConditionalGeneration
    kwargs = {"torch_dtype": torch.float16 if is_llava else torch.bfloat16,
              "attn_implementation": spec.model.get("attn_implementation", "sdpa")}
    if is_llava:
        model = cls.from_pretrained(spec.model["pretrained"], **kwargs).to(device).eval()
    else:
        model = cls.from_pretrained(spec.model["pretrained"], device_map="auto" if device == "cuda" else device,
                                   **kwargs).eval()
    processor_kwargs = {}
    if spec.model.get("max_pixels"):
        processor_kwargs["max_pixels"] = int(spec.model["max_pixels"])
    processor = AutoProcessor.from_pretrained(spec.model["pretrained"], **processor_kwargs)
    return model, processor


def build_inputs(processor, model, image, question: str) -> dict:
    import torch
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = source.convert("RGB")
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    reference = next(model.parameters())
    return {key: (value.to(device=reference.device,
                  dtype=reference.dtype if value.is_floating_point() else torch.long)
                  if isinstance(value, torch.Tensor) else value)
            for key, value in inputs.items()}


def gqa_prompt(question: str) -> str:
    """The lmms-eval v0.7.1 GQA default for llava_hf and qwen2_5_vl.

    Source: lmms_eval/tasks/gqa/gqa.yaml at tag v0.7.1. Its qwen_vl
    exception is a different model class, not this release's qwen2_5_vl.
    """
    return question + "\nAnswer the question using a single word or phrase."
