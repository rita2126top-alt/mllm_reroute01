"""Two-stage residual distillation without changing frozen backbone weights."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint


@dataclass
class TeacherLayer:
    visual_input: torch.Tensor
    residual: torch.Tensor


@dataclass
class LossBundle:
    residual: torch.Tensor
    direction: torch.Tensor
    reactivation: torch.Tensor
    freshness: torch.Tensor
    total: torch.Tensor
    ghost_count: int
    direction_count: int
    candidate_count: int
    reactivation_count: int

    def report(self) -> dict:
        return {key: float(getattr(self, key).detach().cpu())
                for key in ("residual", "direction", "reactivation", "freshness", "total")} | {
                    key: getattr(self, key) for key in ("ghost_count", "direction_count", "candidate_count", "reactivation_count")}


def compute_losses(trace: dict, decisions: dict, teacher: dict[int, TeacherLayer], *,
                   stage="rollout", ablation="full", eps=1e-6, zero_anchor=None) -> LossBundle:
    """Use the SAME student forward's future decisions as detached BCE labels.

    Each term is averaged over its valid token events, not over layers.
    Freshness observes decision-layer INPUT, before the reactivated Full block.
    """
    if stage not in ("warmup", "rollout"):
        raise ValueError(f"Unknown stage: {stage}")
    if zero_anchor is None:
        zero_anchor = next((item.predicted_residual for item in trace.values()), torch.zeros(()))
    zero = zero_anchor.float().sum() * 0.0
    residual, direction, reactivation, freshness = zero, zero, zero, zero
    ghosts = directions = candidates = events = 0
    for layer_idx, item in sorted(trace.items()):
        if layer_idx not in teacher:
            raise ValueError(f"Teacher did not record layer {layer_idx}")
        ghost_ids = item.ghost_ids.long()
        if ghost_ids.numel():
            prediction = item.predicted_residual.float()
            target = teacher[layer_idx].residual.to(device=prediction.device, dtype=torch.float32).index_select(1, ghost_ids)
            if prediction.shape != target.shape:
                raise ValueError("Ghost prediction/teacher residual shapes differ")
            residual = residual + (prediction - target).square().mean(-1).sum()
            ghosts += target.shape[0] * target.shape[1]
            valid = target.norm(dim=-1) >= eps
            cosine = (prediction * target).sum(-1) / (prediction.norm(dim=-1).clamp_min(eps) * target.norm(dim=-1).clamp_min(eps))
            direction = direction + (1.0 - cosine)[valid].sum()
            directions += int(valid.sum().item())
        if item.next_decision is not None and item.skip_ids.numel() and ablation != "score":
            if item.react_logits is None:
                raise ValueError("Learned selector did not record reactivation logits")
            if item.next_decision not in decisions:
                raise ValueError("Missing future decision from this student forward")
            logits = item.react_logits.float()
            labels = decisions[item.next_decision].selected_mask.detach().to(logits.device).index_select(1, item.skip_ids.long()).float()
            reactivation = reactivation + F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            candidates += labels.numel()
        if stage == "rollout" and ablation != "no_fresh" and layer_idx in decisions and layer_idx - 1 in trace:
            previous = trace[layer_idx - 1].active_ids
            returned = item.active_ids[~torch.isin(item.active_ids, previous)]
            if returned.numel():
                current = item.visual_input.float().index_select(1, returned.long())
                target = teacher[layer_idx].visual_input.to(current.device, torch.float32).index_select(1, returned.long())
                freshness = freshness + (current - target).square().mean(-1).sum()
                events += current.shape[0] * current.shape[1]
    residual = residual / max(1, ghosts)
    direction = direction / max(1, directions)
    reactivation = reactivation / max(1, candidates)
    freshness = freshness / max(1, events)
    total = residual + 0.1 * direction + 0.1 * reactivation + freshness
    return LossBundle(residual, direction, reactivation, freshness, total,
                      ghosts, directions, candidates, events)


def freeze_backbone(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def make_optimizer(modules, drop_layers) -> torch.optim.Optimizer:
    """Only matrices decay; queries, vector heads and scalar biases do not."""
    if not drop_layers:
        raise ValueError("Ghost training needs a nonempty decision schedule")
    first, last = min(drop_layers), max(drop_layers)
    active_groups = {layer // modules.config.share_every for layer in range(first, last)}
    decay, no_decay = [], []
    for name, parameter in modules.named_parameters():
        # Modules are block shared; no Ghost is called before first / at last decision.
        parts = name.split(".")
        numeric = next((int(part) for part in parts if part.isdigit()), None)
        used = parameter.requires_grad and (numeric is None or numeric in active_groups)
        parameter.requires_grad_(used)
        if not used:
            continue
        is_query = "prototype_queries" in name.lower() or any(part in ("gate", "reactivation") for part in parts)
        (decay if parameter.ndim >= 2 and not is_query else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("No Ghost parameters are used by this schedule")
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=1e-4, betas=(0.9, 0.999), eps=1e-8)


@contextmanager
def dense_teacher_capture(ctx):
    """Temporarily execute dense layers of the SAME backbone under no_grad.

    CPU copies keep teacher supervision out of the student activation budget.
    The outer prefill hook still discovers/reset the current image range.
    The finally block restores every patched forward even if a sample fails.
    """
    saved = [layer.forward for layer in ctx.layers]
    hooks, records = [], {}
    try:
        for index, layer in enumerate(ctx.layers):
            layer.forward = ctx.original_forwards[index]

            def capture(module, args, kwargs, output, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                result = output[0] if isinstance(output, (tuple, list)) else output
                start, end = ctx.visual_token_range
                if end <= start or hidden.shape[1] < end:
                    raise ValueError("Dense teacher could not locate the prompt visual tokens")
                inputs = hidden[:, start:end].detach().float()
                residual = result[:, start:end].detach().float() - inputs
                records[index] = TeacherLayer(inputs.cpu(), residual.cpu())

            hooks.append(layer.register_forward_hook(capture, with_kwargs=True))
        with torch.no_grad():
            yield records
    finally:
        for hook in hooks:
            hook.remove()
        for layer, forward in zip(ctx.layers, saved):
            layer.forward = forward


def forward_training_sample(model, ctx, inputs: dict, *, stage="rollout", ablation="full"):
    if "labels" in inputs:
        raise ValueError("Cold/Ghost training only accepts image/question prompt inputs; answers/labels are forbidden")
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    with dense_teacher_capture(ctx) as teacher:
        output = model(**kwargs)
        del output
    ctx.apply_updates = stage == "rollout"
    output = model(**kwargs)
    del output
    losses = compute_losses(ctx.trace, ctx.decisions, teacher, stage=stage, ablation=ablation)
    return losses, teacher


class TwoStageTrainer:
    """Deterministic one-epoch warmup followed by one-epoch rollout.

    Resume checkpoints are written ONLY at accumulation boundaries, so there
    are no unsaved pending gradients. ``sample_forward(index, stage)`` returns
    a LossBundle; this callback also enables a real CPU synthetic exercise.
    """

    def __init__(self, modules, metadata, output, *, gradient_accumulation=16,
                 smoke=False, steps_per_stage=None, save_every=50):
        if gradient_accumulation < 1 or save_every < 1:
            raise ValueError("Accumulation and save interval must be positive")
        if steps_per_stage is not None and (not smoke or steps_per_stage < 1):
            raise ValueError("Shortened stages require --smoke and a positive --steps-per-stage")
        self.modules, self.metadata, self.output = modules, metadata, output
        self.accumulation, self.save_every = gradient_accumulation, save_every
        self.steps_per_stage = steps_per_stage
        self.optimizer = make_optimizer(modules, metadata["drop_layers"])
        self.state = {"stage_index": 0, "sample_cursor": 0, "optimizer_steps": 0,
                      "complete": False, "smoke": bool(smoke), "training_seed": 42,
                      "gradient_accumulation": gradient_accumulation, "steps_per_stage": steps_per_stage}

    def resume(self, path):
        result = load_checkpoint(path, self.modules, expected_metadata=self.metadata,
                                 optimizer=self.optimizer, require_complete=False)
        saved = result["training_state"]
        for key in ("smoke", "training_seed", "gradient_accumulation", "steps_per_stage"):
            if saved.get(key) != self.state[key]:
                raise ValueError(f"Resume training setting mismatch: {key}")
        self.state = saved
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])

    def save(self):
        self.state["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            self.state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        save_checkpoint(self.output, self.modules, self.metadata, optimizer=self.optimizer, training_state=self.state)

    def run(self, train_size, sample_forward, *, max_steps=None, on_step=None, validate=None):
        if train_size < 1 or (max_steps is not None and max_steps < 1):
            raise ValueError("Training size and maximum optimizer steps must be positive")
        if self.state["complete"]:
            return self.state
        self.modules.train()
        for stage_index in range(self.state["stage_index"], 2):
            stage = ("warmup", "rollout")[stage_index]
            order = list(range(train_size))
            random.Random(42 + stage_index).shuffle(order)
            if self.steps_per_stage is not None:
                order = order[:self.steps_per_stage * self.accumulation]
            start = self.state["sample_cursor"] if self.state["stage_index"] == stage_index else 0
            for cursor in range(start, len(order), self.accumulation):
                if max_steps is not None and self.state["optimizer_steps"] >= max_steps:
                    self.save()
                    return self.state
                batch = order[cursor:cursor + self.accumulation]
                self.optimizer.zero_grad(set_to_none=True)
                reports, has_grad = [], False
                for index in batch:
                    losses = sample_forward(index, stage)
                    if not torch.isfinite(losses.total):
                        raise FloatingPointError(f"Non-finite training loss at {stage}, sample {index}")
                    reports.append(losses.report())
                    if losses.total.requires_grad:
                        (losses.total / len(batch)).backward()
                        has_grad = True
                if has_grad:
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                self.state.update(stage_index=stage_index, sample_cursor=cursor + len(batch),
                                  optimizer_steps=self.state["optimizer_steps"] + 1)
                if on_step:
                    on_step(dict(self.state), stage, reports)
                if self.state["optimizer_steps"] % self.save_every == 0:
                    self.save()
            if validate:
                self.state[f"validation_{stage}"] = validate(stage)
            self.state.update(stage_index=stage_index + 1, sample_cursor=0)
            self.save()
        self.state["complete"] = True
        self.save()
        return self.state


def mean_reports(reports: list[dict]) -> dict:
    """Report token-weighted losses on independent validation samples."""
    if not reports:
        raise ValueError("Validation dataset is empty")
    counts = {key: sum(row[key] for row in reports) for key in
              ("ghost_count", "direction_count", "candidate_count", "reactivation_count")}
    terms = {"residual": "ghost_count", "direction": "direction_count",
             "reactivation": "candidate_count", "freshness": "reactivation_count"}
    result = {term: sum(row[term] * row[count] for row in reports) / max(1, counts[count])
              for term, count in terms.items()}
    result["total"] = result["residual"] + 0.1 * result["direction"] + 0.1 * result["reactivation"] + result["freshness"]
    return result | counts | {"samples": len(reports)}
