"""Bundled-cohort efficiency measurements with explicit FLOP coverage limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from .config import ABLATIONS, ROOT, load_experiment
from .evaluation import new_output_dir, write_json


def walk_kv(cache) -> dict:
    """Read actual post-prefill cache tensors (not projected theoretical sizes)."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        items = zip(cache.key_cache, cache.value_cache)
    elif hasattr(cache, "layers"):
        items = ((layer.keys, layer.values) for layer in cache.layers)
    else:
        items = ((layer[0], layer[1]) for layer in cache)
    per_layer, byte_count, tokens = [], 0, 0
    for key, value in items:
        if key is None or value is None:
            per_layer.append(0)
            continue
        per_layer.append(int(key.shape[-2]))
        tokens += int(key.shape[-2] + value.shape[-2])
        byte_count += key.numel() * key.element_size() + value.numel() * value.element_size()
    return {"kv_bytes": byte_count, "kv_k_plus_v_tokens": tokens,
            "per_layer_kv_sequence_lengths": per_layer}


def _cohort():
    base = ROOT / "bench_data"
    samples = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if len(samples) != 3:
        raise ValueError("The release efficiency cohort must contain the original three samples")
    for sample in samples:
        if not (base / sample["image"]).is_file():
            raise FileNotFoundError(base / sample["image"])
        if sample["question"] != "Describe this image in detail.":
            raise ValueError("Efficiency prompt differs from the release cohort")
    return base, samples


def _auxiliary(ctx):
    counters = dict(getattr(ctx, "counters", {})) if ctx is not None else {}
    method = getattr(ctx, "auxiliary_state_bytes", None)
    return {"auxiliary_state_bytes_at_prefill_end": int(method()) if method else _original_auxiliary_bytes(ctx),
            "auxiliary_state_bytes_peak": getattr(ctx, "peak_auxiliary_state_bytes", None),
            "gc_counters": counters,
            "gc_matrix_flops_analytic": 2 * counters["estimated_macs"]
                if "estimated_macs" in counters else 0}


def _timed(torch, call):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def _summary(values):
    import numpy as np
    return {"n": len(values), "mean": statistics.mean(values),
            "p50_median": statistics.median(values),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90))}


def run_efficiency(config, *, checkpoint=None, ablation="full", mode="profile",
                   n_passes=None, n_warmup_passes=None, n_decode_tokens=64):
    import torch
    from .loading import build_inputs, load_backbone, patch_experiment, validate_checkpoint
    if mode not in ("profile", "benchmark"):
        raise ValueError("mode must be profile or benchmark")
    if not torch.cuda.is_available():
        raise RuntimeError("The release efficiency protocol requires a CUDA GPU")
    n_passes = (1 if mode == "profile" else 5) if n_passes is None else n_passes
    n_warmup_passes = (0 if mode == "profile" else 2) if n_warmup_passes is None else n_warmup_passes
    if n_passes < 1 or n_warmup_passes < 0 or n_decode_tokens < 1:
        raise ValueError("passes/decode tokens must be positive and warmup non-negative")
    spec = load_experiment(config)
    if checkpoint:
        validate_checkpoint(spec, checkpoint, ablation)
    model, processor = load_backbone(spec)
    ctx = patch_experiment(model, spec, checkpoint, ablation)
    base, samples = _cohort()
    inputs = [build_inputs(processor, model, base / sample["image"], sample["question"])
              for sample in samples]
    generation = dict(min_new_tokens=n_decode_tokens, max_new_tokens=n_decode_tokens,
                      do_sample=False, use_cache=True, return_dict_in_generate=True)
    with torch.inference_mode():
        for _ in range(n_warmup_passes):
            for batch in inputs:
                if mode == "benchmark":
                    warm = model.generate(**batch, **generation)
                else:
                    warm = model(**batch, use_cache=True)
                del warm
            torch.cuda.empty_cache()
        records = []
        for pass_index in range(n_passes):
            for sample, batch in zip(samples, inputs):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                before_allocated = torch.cuda.memory_allocated()
                if mode == "profile":
                    # Dispatch-level formulas count the real executed linear, conv,
                    # explicit score QK and fused SDPA QK/AV matrix operations.
                    with registered_flop_counter() as counter:
                        result = model(**batch, use_cache=True)
                    torch.cuda.synchronize()
                    registered_flops = int(counter.get_total_flops())
                    record = {"model_registered_flops": registered_flops,
                              "model_registered_tflops": registered_flops / 1e12,
                              "registered_flops_by_operator": {
                                  str(op): int(count) for op, count in
                                  counter.get_flop_counts().get("Global", {}).items()},
                              "flops_coverage": FLOPS_COVERAGE}
                else:
                    result, prefill_ms = _timed(torch, lambda: model(**batch, use_cache=True))
                    record = {"prefill_ms": prefill_ms}
                record.update(walk_kv(result.past_key_values))
                record.update(_auxiliary(ctx))
                record["prefill_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["prefill_peak_increment_bytes"] = torch.cuda.max_memory_allocated() - before_allocated
                del result
                if mode == "benchmark":
                    torch.cuda.reset_peak_memory_stats()
                    generated, e2e_ms = _timed(torch, lambda: model.generate(**batch, **generation))
                    record["e2e_ms"] = e2e_ms
                    record["decode_ms_per_token_estimate"] = (e2e_ms - record["prefill_ms"]) / n_decode_tokens
                    record["e2e_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    if pass_index == 0:
                        ids = generated.sequences[0, batch["input_ids"].shape[1]:]
                        tokenizer = getattr(processor, "tokenizer", processor)
                        record["response"] = tokenizer.decode(ids, skip_special_tokens=True)
                    del generated
                record.update({"sample_id": sample["sample_id"], "pass": pass_index,
                               "prefill_tokens": int(batch["input_ids"].shape[1])})
                records.append(record)
    metrics = ("model_registered_flops", "model_registered_tflops", "gc_matrix_flops_analytic", "kv_bytes",
               "auxiliary_state_bytes_at_prefill_end", "prefill_peak_allocated_bytes",
               "prefill_ms", "e2e_ms", "decode_ms_per_token_estimate")
    aggregates = {metric: _summary([row[metric] for row in records if metric in row])
                  for metric in metrics if any(metric in row for row in records)}
    return {"config": spec.config_name, "checkpoint": str(checkpoint) if checkpoint else None,
            "ablation": ablation, "mode": mode, "settings": {"n_passes": n_passes,
                "n_warmup_passes": n_warmup_passes,
                "n_decode_tokens": n_decode_tokens if mode == "benchmark" else None},
            "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                         "cuda": torch.version.cuda},
            "flops_note": "GC analytic matrix FLOPs are a subset of model_registered_flops: do not add them again. Registered FLOPs include fused attention matrix products but exclude unregistered elementwise/nonlinear/selection operations. Report this convention alongside TFLOPs. All latency measurements execute real scoring and Ghost operations; CPU preprocessing is excluded.",
            "per_sample": records, "aggregate": aggregates}


def cli(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Cold/Ghost {mode} using the original 3-image cohort")
    parser.add_argument("--config", required=True)
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--checkpoint", type=Path)
    arm.add_argument("--original", action="store_true")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--n-passes", type=int, default=1 if mode == "profile" else 5)
    parser.add_argument("--n-warmup-passes", type=int, default=0 if mode == "profile" else 2)
    if mode == "benchmark":
        parser.add_argument("--n-decode-tokens", type=int, default=64)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec = load_experiment(args.config)
    if args.checkpoint and not spec.is_reroute:
        parser.error("Only Reroute configurations support Ghost")
    if args.n_passes < 1 or args.n_warmup_passes < 0 or getattr(args, "n_decode_tokens", 64) < 1:
        parser.error("Invalid repetition/decode count")
    if args.original and args.ablation != "full":
        parser.error("Ablations require a Ghost checkpoint")
    if args.dry_run:
        print(json.dumps({**vars(args), "config": spec.config_name, "mode": mode}, default=str, indent=2))
        return 0
    output = new_output_dir(args.out)
    try:
        result = run_efficiency(spec.config_name, checkpoint=args.checkpoint,
            ablation=args.ablation, mode=mode, n_passes=args.n_passes,
            n_warmup_passes=args.n_warmup_passes,
            n_decode_tokens=getattr(args, "n_decode_tokens", 64))
        write_json(output / "results.json", result)
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return 0


FLOPS_COVERAGE = (
    "PyTorch FlopCounterMode registered-operation convention, 2 FLOPs per MAC: "
    "linear/matmul/bmm, convolution, explicit decision-score QK, fused SDPA QK and AV, "
    "including Ghost operations. Excludes unregistered elementwise arithmetic, bias "
    "addition, norms, softmax/SiLU/sigmoid, RoPE elementwise work, sorting, selection "
    "and memory movement. Attention uses dense QK/AV arithmetic-equivalent formulas "
    "even for causal masks. Inspect registered_flops_by_operator for actual coverage."
)


def registered_flop_counter():
    """Count fused SDPA as well as explicit matrix operations without double-counting.

    PyTorch 2.11 registers CUDA flash/efficient/cuDNN SDPA. Register the CPU
    flash packet locally too, so CPU verification uses identical QK+AV arithmetic.
    No global PyTorch registries or model attention backends are modified.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count
    def cpu_sdpa(query_shape, key_shape, value_shape, *args, **kwargs):
        return sdpa_flop_count(query_shape, key_shape, value_shape)
    cpu_flash = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    mapping = {cpu_flash: cpu_sdpa} if cpu_flash is not None else {}
    return FlopCounterMode(display=False, custom_mapping=mapping)

def _original_auxiliary_bytes(ctx) -> int:
    """Count retained release context tensors too, excluding frozen module weights."""
    if ctx is None:
        return 0
    import dataclasses
    import torch
    storages = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))
    for name in ("attn_weights_cache", "routing_log", "_stage_deferred_hidden",
                 "_stage_deferred_indices", "_stage_compact_kept_indices",
                 "_compact_position_embeddings", "_compact_position_ids"):
        visit(getattr(ctx, name, None))
    router = getattr(ctx, "router", None)
    visit(getattr(router, "_stage_decisions", None))
    visit(getattr(router, "_cached_decision", None))
    return sum(storages.values())
