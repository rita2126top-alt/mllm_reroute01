"""Additive compact/stagewise Cold-Ghost patches; original files stay intact."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import functools
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from models.patching import _capture_attention_weights, get_visual_token_finder
from models.router import PDropRouter, FastVRouter, RoutingDecision
from .ghost import GhostModules


@dataclass
class LayerTrace:
    visual_input: Tensor
    active_ids: Tensor
    skip_ids: Tensor
    ghost_ids: Tensor
    predicted_residual: Tensor
    react_logits: Tensor | None
    age: Tensor
    next_decision: int | None
    active_residual: Tensor
    decision_scores: Tensor | None


@dataclass
class GCContext:
    router: Any
    action: str
    ghost_modules: GhostModules
    model_family: str
    collect_trace: bool = False
    apply_updates: bool = True
    checkpoint_layers: bool = False
    detach_trace: bool = False
    visual_token_range: tuple[int, int] = (0, 0)
    trace: dict[int, LayerTrace] = field(default_factory=dict)
    decisions: dict[int, RoutingDecision] = field(default_factory=dict)
    routing_log: dict[int, RoutingDecision] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    layers: list = field(default_factory=list)
    original_forwards: dict[int, Any] = field(default_factory=dict)
    age: Tensor | None = None
    attn_weights_cache: Tensor | None = None
    _layer_modules: dict = field(default_factory=dict)
    _kept: Tensor | None = None
    _deferred: Tensor | None = None
    _deferred_hidden: Tensor | None = None
    _full_length: int = 0
    _active_decision: RoutingDecision | None = None
    _prefill_active: bool = False
    peak_auxiliary_state_bytes: int = 0

    @property
    def _model_family(self):
        return self.model_family

    @property
    def decision_layers(self) -> list[int]:
        if isinstance(self.router, PDropRouter):
            return list(self.router.drop_layers)
        return [self.router.scoring_layer]

    def reset(self, visual_token_range: tuple[int, int] = (0, 0)):
        self.visual_token_range = visual_token_range
        self.router.reset()
        self.trace.clear()
        self.decisions.clear()
        self.routing_log.clear()
        self.counters = dict(ghost_token_layers=0, skip_token_layers=0,
                             full_token_layers=0, prototype_layers=0,
                             estimated_macs=0)
        self.age = None
        self.attn_weights_cache = None
        self._kept = self._deferred = self._deferred_hidden = None
        self._full_length = 0
        self._active_decision = None
        self._prefill_active = visual_token_range[1] > visual_token_range[0]
        self.peak_auxiliary_state_bytes = 0

    def next_decision(self, layer_idx: int) -> int | None:
        return next((d for d in self.decision_layers if d > layer_idx), None)

    def auxiliary_state_bytes(self) -> int:
        """Live auxiliary tensor storage, excluding parameters and KV cache."""
        seen = set()
        def size(value):
            if isinstance(value, Tensor):
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr())
                if key in seen:
                    return 0
                seen.add(key)
                return storage.nbytes()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(size(v) for v in value)
            if isinstance(value, (LayerTrace, RoutingDecision)):
                return sum(size(getattr(value, f.name)) for f in fields(value))
            return 0
        return size((self.age, self.attn_weights_cache, self._kept,
                     self._deferred, self._deferred_hidden,
                     self.trace, self.decisions, self.routing_log))

    def update_memory_peak(self):
        self.peak_auxiliary_state_bytes = max(self.peak_auxiliary_state_bytes,
                                               self.auxiliary_state_bytes())


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def _gather_positions(position_ids, position_embeddings, kept):
    positions = None if position_ids is None else position_ids.index_select(-1, kept)
    embeddings = None if position_embeddings is None else tuple(
        embedding.index_select(-2, kept) for embedding in position_embeddings)
    return positions, embeddings


def _run_original(ctx, original_forward, hidden_states, kwargs):
    # Only this pure call is replayed. Routing/age/deferred mutations stay out.
    if ctx.checkpoint_layers and torch.is_grad_enabled():
        if kwargs.get("use_cache", False):
            raise ValueError("Activation checkpointing requires use_cache=False")
        return checkpoint(original_forward, hidden_states,
                          use_reentrant=False, **kwargs)
    return original_forward(hidden_states, **kwargs)


def _record(ctx, layer_idx, trace):
    if ctx.collect_trace:
        if ctx.detach_trace:
            trace = LayerTrace(**{f.name: (getattr(trace, f.name).detach()
                                          if isinstance(getattr(trace, f.name), Tensor)
                                          else getattr(trace, f.name)) for f in fields(trace)})
        ctx.trace[layer_idx] = trace


def _reconstruct(ctx, compact):
    full = compact.new_zeros((1, ctx._full_length, compact.shape[-1]))
    return full.index_copy(1, ctx._kept, compact).index_copy(1, ctx._deferred, ctx._deferred_hidden)


def _dense_step(ctx, layer_idx, original, hidden, kwargs):
    start, end = ctx.visual_token_range
    visual = hidden[:, start:end]
    output = _run_original(ctx, original, hidden, kwargs)
    ids = torch.arange(end - start, device=hidden.device)
    empty = ids[:0]
    before_age = ctx.age
    ctx.age = torch.zeros_like(ctx.age)
    ctx.counters["full_token_layers"] += end - start
    if ctx.collect_trace:
        _record(ctx, layer_idx, LayerTrace(
            visual, ids, empty, empty, visual.new_empty((1, 0, visual.shape[-1]), dtype=torch.float32),
            None, before_age, ctx.next_decision(layer_idx),
            _hidden(output)[:, start:end].float() - visual.float(), None))
    return output


def _routed_step(ctx, layer_idx, original, hidden, kwargs, stage_input):
    start, end = ctx.visual_token_range
    n = end - start
    decision_layer = ctx.router.is_decision_layer(layer_idx)
    full_input = None
    if not stage_input or decision_layer:
        full_input = _reconstruct(ctx, hidden) if stage_input else hidden
        if decision_layer:
            _capture_attention_weights(layer_idx, ctx, full_input,
                                       kwargs.get("attention_mask"), kwargs.get("position_embeddings"))
        decision = ctx.router.compute_scores(
            layer_idx=layer_idx, attn_weights=ctx.attn_weights_cache,
            hidden_states=full_input, visual_token_range=ctx.visual_token_range)
        ctx._active_decision = decision
        if decision_layer:
            ctx.decisions[layer_idx] = decision
        keep_mask = torch.ones(full_input.shape[1], dtype=torch.bool, device=hidden.device)
        skip_ids = (~decision.selected_mask[0]).nonzero(as_tuple=True)[0]
        keep_mask[skip_ids + start] = False
        kept = keep_mask.nonzero(as_tuple=True)[0]
        deferred = (~keep_mask).nonzero(as_tuple=True)[0]
        compact_input = full_input.index_select(1, kept)
        skip_input = full_input.index_select(1, deferred)
        ctx._full_length = full_input.shape[1]
    else:
        decision = ctx._active_decision
        kept, deferred = ctx._kept, ctx._deferred
        skip_ids = deferred - start
        compact_input, skip_input = hidden, ctx._deferred_hidden
    ctx.routing_log[layer_idx] = decision
    active_offsets = ((kept >= start) & (kept < end)).nonzero(as_tuple=True)[0]
    active_ids = kept.index_select(0, active_offsets) - start
    active_input = compact_input.index_select(1, active_offsets)
    compact_positions, compact_embeddings = _gather_positions(
        kwargs.get("position_ids"), kwargs.get("position_embeddings"), kept)
    compact_kwargs = dict(kwargs, attention_mask=None,
                          position_ids=compact_positions, position_embeddings=compact_embeddings)
    if isinstance(compact_kwargs.get("cache_position"), Tensor):
        compact_kwargs["cache_position"] = compact_kwargs["cache_position"].index_select(-1, kept)
    compact_output = _run_original(ctx, original, compact_input, compact_kwargs)
    compact_hidden = _hidden(compact_output)
    next_decision = ctx.next_decision(layer_idx)
    needs_residual = ctx.collect_trace or (next_decision is not None and skip_ids.numel() and ctx.ghost_modules.config.ghost_budget)
    active_residual = (compact_hidden.index_select(1, active_offsets).float() - active_input.float()
                       if needs_residual else hidden.new_empty((1, 0, hidden.shape[-1]), dtype=torch.float32))
    before_age = ctx.age
    empty = skip_ids[:0]
    ghost_ids = empty
    prediction = hidden.new_empty((1, 0, hidden.shape[-1]), dtype=torch.float32)
    logits = None
    skip_output = skip_input
    if next_decision is not None and skip_ids.numel() and ctx.ghost_modules.config.ghost_budget:
        threshold = decision.scores.index_select(1, active_ids).amin(dim=1)
        result = ctx.ghost_modules(
            layer_idx, active_input=active_input, active_residual=active_residual,
            skip_input=skip_input, skip_ids=skip_ids,
            skip_scores=decision.scores.index_select(1, skip_ids), threshold=threshold,
            skip_age=before_age.index_select(0, skip_ids), next_decision=next_decision)
        ghost_ids, prediction, logits = result.ghost_ids, result.predicted_residual, result.react_logits
        if ctx.apply_updates:
            changed = (skip_input.index_select(1, result.ghost_offsets).float() + prediction).to(hidden.dtype)
            skip_output = skip_input.index_copy(1, result.ghost_offsets, changed)
        ctx.counters["estimated_macs"] += result.estimated_macs
        ctx.counters["prototype_layers"] += int(result.prototypes is not None)
    if ctx.collect_trace:
        if full_input is not None:
            visual_input = full_input[:, start:end]
        else:
            visual_input = hidden.new_zeros((1, n, hidden.shape[-1]))
            visual_input = visual_input.index_copy(1, active_ids, active_input).index_copy(1, skip_ids, skip_input)
        _record(ctx, layer_idx, LayerTrace(
            visual_input, active_ids, skip_ids, ghost_ids, prediction, logits,
            before_age, next_decision, active_residual, decision.scores))
    ctx.age = (before_age + 1).index_fill(0, active_ids, 0)
    ctx.counters["full_token_layers"] += active_ids.numel()
    ctx.counters["skip_token_layers"] += skip_ids.numel()
    ctx.counters["ghost_token_layers"] += ghost_ids.numel()
    if ctx.action == "compact_route_stagewise":
        ctx._kept, ctx._deferred, ctx._deferred_hidden = kept, deferred, skip_output
        return compact_output
    full_output = full_input.index_copy(1, kept, compact_hidden).index_copy(1, deferred, skip_output)
    return _replace_hidden(compact_output, full_output)


def _make_forward(ctx, layer_idx, original):
    @functools.wraps(original)
    def forward(hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        kwargs = dict(kwargs, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_values=past_key_values, use_cache=use_cache,
                      position_embeddings=position_embeddings)
        if hidden_states.shape[0] != 1:
            raise ValueError("Cold/Ghost currently requires batch size 1")
        start, end = ctx.visual_token_range
        if not ctx._prefill_active or hidden_states.shape[1] == 1 or end <= start:
            if end > start and hidden_states.shape[1] == 1 and layer_idx >= ctx.decision_layers[0]:
                kwargs["attention_mask"] = None
            return original(hidden_states, **kwargs)
        if ctx.age is None:
            ctx.age = torch.zeros(end - start, dtype=torch.long, device=hidden_states.device)
        if not ctx.router.should_route(layer_idx):
            result = _dense_step(ctx, layer_idx, original, hidden_states, kwargs)
        else:
            stage_input = ctx.action == "compact_route_stagewise" and ctx._kept is not None
            result = _routed_step(ctx, layer_idx, original, hidden_states, kwargs, stage_input)
        ctx.update_memory_peak()
        return result
    return forward


def patch_model_with_cold_ghost(model, router, action="compact_route", model_family="llava",
                               ghost_modules=None, collect_trace=False, apply_updates=True,
                               checkpoint_layers=False, detach_trace=False) -> GCContext:
    if action not in {"compact_route", "compact_route_stagewise"}:
        raise ValueError("Cold/Ghost only extends recoverable compact routing")
    if not isinstance(router, (PDropRouter, FastVRouter)):
        raise TypeError("Use the original project PDropRouter or FastVRouter")
    if isinstance(router, PDropRouter) and router.monotonic:
        raise ValueError("Cold/Ghost requires the original non-monotonic Reroute setting")
    if hasattr(model, "_cold_ghost_context") or hasattr(model, "_routing_pre_hook_handle"):
        raise ValueError("Model is already patched; unpatch before installing Cold/Ghost")
    layers = list(model.model.language_model.layers)
    if ghost_modules is None:
        hidden_size = layers[0].self_attn.q_proj.in_features
        ghost_modules = GhostModules(hidden_size, len(layers)).to(next(model.parameters()).device)
    if ghost_modules.num_layers != len(layers):
        raise ValueError("Ghost module layer count does not match the backbone")
    ctx = GCContext(router, action, ghost_modules, model_family, collect_trace,
                    apply_updates, checkpoint_layers, detach_trace)
    ctx.layers = layers
    ctx._layer_modules = dict(enumerate(layers))
    ctx.original_forwards = {i: layer.forward for i, layer in enumerate(layers)}
    decisions = ctx.decision_layers
    if not decisions or decisions != sorted(set(decisions)) or decisions[0] < 0 or decisions[-1] >= len(layers):
        raise ValueError("Decision layers must be sorted, unique, and inside the backbone")
    ctx.reset()
    for i, layer in enumerate(layers):
        layer.forward = _make_forward(ctx, i, ctx.original_forwards[i])
    finder = get_visual_token_finder(model_family)
    def pre_forward(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        if input_ids.shape[0] != 1:
            raise ValueError("Cold/Ghost currently requires batch size 1")
        if input_ids.shape[-1] > 1:
            mask = kwargs.get("attention_mask")
            if isinstance(mask, Tensor) and mask.ndim == 2 and not bool(mask.all()):
                raise ValueError("Cold/Ghost currently requires unpadded batch-size-one prompts")
            vis_range = finder(module, input_ids, pixel_values=kwargs.get("pixel_values"))
            ctx.reset(vis_range)
        else:
            ctx._prefill_active = False
    model._cold_ghost_pre_hook_handle = model.register_forward_pre_hook(pre_forward, with_kwargs=True)
    model._cold_ghost_context = ctx
    return ctx


def unpatch_model_with_cold_ghost(model):
    ctx = getattr(model, "_cold_ghost_context", None)
    if ctx is None:
        return
    for i, layer in enumerate(ctx.layers):
        layer.forward = ctx.original_forwards[i]
    model._cold_ghost_pre_hook_handle.remove()
    del model._cold_ghost_pre_hook_handle
    del model._cold_ghost_context
