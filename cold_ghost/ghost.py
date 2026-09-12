"""FP32 residual predictors for the additive Cold/Ghost extension.

This module never makes a Full-token routing decision.  It only ranks and
updates candidates supplied by the unmodified project router.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GhostConfig:
    bottleneck_dim: int = 32
    num_prototypes: int = 8
    ghost_budget: int = 128
    share_every: int = 4
    age_cap: int = 8
    eps: float = 1e-6
    ablation: str = "full"

    def __post_init__(self):
        for name in ("bottleneck_dim", "num_prototypes", "share_every", "age_cap"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.ghost_budget < 0 or self.eps <= 0:
            raise ValueError("ghost_budget must be nonnegative and eps positive")
        if self.ablation not in {"full", "self", "context", "score", "no_fresh"}:
            raise ValueError(f"Unknown ablation: {self.ablation}")


@dataclass
class GhostResult:
    ghost_offsets: Tensor
    ghost_ids: Tensor
    predicted_residual: Tensor
    react_logits: Tensor | None
    prototypes: Tensor | None
    estimated_macs: int


class _GhostBlock(nn.Module):
    def __init__(self, hidden_size: int, config: GhostConfig):
        super().__init__()
        d, b, m = hidden_size, config.bottleneck_dim, config.num_prototypes
        self.config = config
        self.down = nn.Linear(d, b, bias=False)
        self.up = nn.Linear(b, d, bias=False)
        self.residual_down = nn.Linear(d, b, bias=False)
        self.prototype_key = nn.Linear(d, b, bias=False)
        self.prototype_queries = nn.Parameter(torch.empty(m, b))
        self.ghost_query = nn.Linear(d, b, bias=False)
        self.ghost_key = nn.Linear(b, b, bias=False)
        self.ghost_value = nn.Linear(b, b, bias=False)
        self.context_up = nn.Linear(b, d, bias=False)
        self.gate = nn.Linear(2 * b + 1, 1)
        self.reactivation = nn.Linear(b + 5, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.up.weight, std=1e-4)
        nn.init.normal_(self.context_up.weight, std=1e-4)
        nn.init.normal_(self.prototype_queries, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.reactivation.weight)
        nn.init.zeros_(self.reactivation.bias)
        if config.ablation == "self":
            for module in (self.residual_down, self.prototype_key, self.ghost_query,
                           self.ghost_key, self.ghost_value, self.context_up):
                module.requires_grad_(False)
            self.prototype_queries.requires_grad_(False)
        if config.ablation == "context":
            self.up.requires_grad_(False)
        if config.ablation == "score":
            self.reactivation.requires_grad_(False)

    def norm(self, values: Tensor) -> Tensor:
        return F.layer_norm(values.float(), (values.shape[-1],), eps=self.config.eps)

    def forward(self, *, active_input: Tensor, active_residual: Tensor,
                skip_input: Tensor, skip_ids: Tensor, skip_scores: Tensor,
                threshold: Tensor, skip_age: Tensor, layer_idx: int,
                num_layers: int, next_decision: int) -> GhostResult:
        cfg = self.config
        b, m, d = cfg.bottleneck_dim, cfg.num_prototypes, skip_input.shape[-1]
        count = min(cfg.ghost_budget, skip_input.shape[1])
        normed_skip = self.norm(skip_input)
        z = self.down(normed_skip)
        age = skip_age.float().clamp(max=cfg.age_cap) / cfg.age_cap
        denom = threshold.float().reshape(-1, 1) + cfg.eps
        scores = skip_scores.float()
        relative_layer = torch.full_like(scores, layer_idx / max(1, num_layers - 1))
        distance = torch.full_like(scores, (next_decision - layer_idx) / max(1, num_layers - 1))
        features = torch.cat((z, (scores / denom).unsqueeze(-1),
                              ((scores - threshold.float().reshape(-1, 1)) / denom).unsqueeze(-1),
                              age.reshape(1, -1, 1), relative_layer.unsqueeze(-1),
                              distance.unsqueeze(-1)), dim=-1)
        logits = None if cfg.ablation == "score" else self.reactivation(features).squeeze(-1)
        priority = scores if logits is None else logits
        # skip_ids arrive in original position order. Stable sort gives the
        # specified original-index tie break without perturbing numeric scores.
        offsets = torch.argsort(priority[0].detach(), descending=True, stable=True)[:count]
        ghost_ids = skip_ids.index_select(0, offsets)
        selected_z = z.index_select(1, offsets)
        selected_norm = normed_skip.index_select(1, offsets)
        context = torch.zeros_like(selected_z)
        prototypes = None
        macs = skip_input.shape[1] * d * b
        if logits is not None:
            macs += skip_input.shape[1] * (b + 5)
        if cfg.ablation != "self" and count:
            codes = self.residual_down(active_residual.float())
            keys = self.prototype_key(self.norm(active_input))
            # (batch, prototypes, active), with independent sample banks.
            weights = torch.softmax(torch.matmul(self.prototype_queries.unsqueeze(0),
                                                  keys.transpose(-1, -2)) / math.sqrt(b), dim=-1)
            prototypes = weights @ codes
            query = self.ghost_query(selected_norm)
            prototype_keys = self.ghost_key(prototypes)
            prototype_values = self.ghost_value(prototypes)
            attention = torch.softmax(query @ prototype_keys.transpose(-1, -2) / math.sqrt(b), dim=-1)
            context = attention @ prototype_values
            a = active_input.shape[1]
            macs += 2 * a * d * b + 2 * a * m * b + 2 * m * b * b + count * d * b + 2 * count * m * b
        update = torch.zeros((1, count, d), device=skip_input.device, dtype=torch.float32)
        if cfg.ablation != "context":
            update = update + self.up(F.silu(selected_z))
            macs += count * d * b
        if cfg.ablation != "self":
            update = update + self.context_up(context)
            macs += count * d * b
        gate_input = torch.cat((selected_z, context, age.index_select(0, offsets).reshape(1, -1, 1)), dim=-1)
        update = torch.sigmoid(self.gate(gate_input)) * update
        macs += count * (2 * b + 1)
        return GhostResult(offsets, ghost_ids, update, logits, prototypes, int(macs))


class GhostModules(nn.Module):
    """Four-layer-shared predictors; state_dict contains only new weights."""

    def __init__(self, hidden_size: int, num_layers: int, config: GhostConfig | None = None):
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = config or GhostConfig()
        self.blocks = nn.ModuleList(_GhostBlock(hidden_size, self.config)
                                    for _ in range(math.ceil(num_layers / self.config.share_every)))
        self.float()

    def group_index(self, layer_idx: int) -> int:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Layer {layer_idx} outside [0, {self.num_layers})")
        return layer_idx // self.config.share_every

    def forward(self, layer_idx: int, **kwargs) -> GhostResult:
        block = self.blocks[self.group_index(layer_idx)]
        if block.down.weight.dtype != torch.float32:
            raise ValueError("Ghost weights must remain FP32; move device without changing dtype")
        skip_input = kwargs["skip_input"]
        if skip_input.shape[0] != 1:
            raise ValueError("Cold/Ghost currently requires batch size 1")
        # Explicitly disable caller autocast: all new projections run in FP32.
        with torch.autocast(device_type=skip_input.device.type, enabled=False):
            return block(layer_idx=layer_idx, num_layers=self.num_layers, **kwargs)

    def active_parameter_count(self, decision_layers: list[int]) -> int:
        if len(decision_layers) < 2 or self.config.ghost_budget == 0:
            return 0
        groups = {self.group_index(l) for l in range(decision_layers[0], decision_layers[-1])}
        return sum(p.numel() for g in groups for p in self.blocks[g].parameters() if p.requires_grad)
