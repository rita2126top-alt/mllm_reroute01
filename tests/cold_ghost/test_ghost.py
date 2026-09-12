import pytest
import torch

from cold_ghost.ghost import GhostConfig, GhostModules


def inputs(dtype=torch.float32):
    return dict(active_input=torch.randn(1, 3, 16, dtype=dtype),
                active_residual=torch.randn(1, 3, 16, dtype=dtype),
                skip_input=torch.randn(1, 5, 16, dtype=dtype),
                skip_ids=torch.tensor([0, 2, 4, 5, 7]),
                skip_scores=torch.tensor([[0.1, 0.3, 0.2, 0.3, 0.0]]),
                threshold=torch.tensor([0.4]), skip_age=torch.tensor([0, 1, 2, 8, 20]),
                next_decision=3)


def test_dimensions_fp32_tie_break_and_shared_groups():
    model = GhostModules(16, 6, GhostConfig(ghost_budget=2))
    result = model(1, **inputs(torch.bfloat16))
    assert result.ghost_ids.tolist() == [0, 2]
    assert result.predicted_residual.shape == (1, 2, 16)
    assert result.predicted_residual.dtype == torch.float32
    assert result.react_logits.shape == (1, 5)
    assert result.prototypes.shape == (1, 8, 32)
    assert model.group_index(0) == model.group_index(3)
    assert model.group_index(4) != model.group_index(3)
    assert result.estimated_macs > 0


@pytest.mark.parametrize("ablation", ["full", "self", "context", "score", "no_fresh"])
def test_ablations_backward_and_score_selector(ablation):
    torch.manual_seed(42)
    model = GhostModules(16, 6, GhostConfig(ghost_budget=2, ablation=ablation))
    data = inputs()
    data["skip_input"].requires_grad_(True)
    data["active_residual"].requires_grad_(True)
    result = model(1, **data)
    target = torch.randn_like(result.predicted_residual)
    loss = (result.predicted_residual - target).square().mean()
    if result.react_logits is not None:
        loss = loss + result.react_logits.square().mean()
    loss.backward()
    assert model.blocks[0].down.weight.grad is not None
    assert torch.isfinite(data["skip_input"].grad).all()
    if ablation == "score":
        assert result.ghost_ids.tolist() == [2, 5]
        assert result.react_logits is None
    if ablation == "self":
        assert result.prototypes is None
        assert data["active_residual"].grad is None
    else:
        assert data["active_residual"].grad is not None


def test_clamps_budget_and_handles_zero_threshold():
    model = GhostModules(16, 6, GhostConfig(ghost_budget=128))
    data = inputs()
    data["threshold"].zero_()
    data["skip_scores"].zero_()
    result = model(1, **data)
    assert result.ghost_ids.numel() == 5
    assert torch.isfinite(result.predicted_residual).all()


def test_context_uses_actual_residual_and_has_no_cross_sample_bank():
    model = GhostModules(16, 6, GhostConfig(ghost_budget=2))
    data = inputs()
    first = model(1, **data)
    data["active_residual"] = data["active_residual"] + 1
    second = model(1, **data)
    assert not torch.equal(first.prototypes, second.prototypes)
    assert not torch.equal(first.predicted_residual, second.predicted_residual)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        GhostConfig(share_every=0)
    with pytest.raises(ValueError):
        GhostConfig(ablation="unknown")
