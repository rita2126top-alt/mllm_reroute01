from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cold_ghost.ghost import GhostConfig, GhostModules
from cold_ghost.training import (TeacherLayer, TwoStageTrainer, compute_losses,
                                  freeze_backbone, make_optimizer, mean_reports)


def trajectory(parameter, *, teacher_zero=False):
    state = torch.zeros(1, 3, 4)
    prediction = parameter.expand(1, 1, 4)
    # Frozen Full layers still propagate the earlier Ghost correction.
    frozen = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad(): frozen.weight.copy_(torch.eye(4) * 2)
    freeze_backbone(frozen)
    later = frozen(state.index_copy(1, torch.tensor([1]), prediction))
    trace = {
        0: SimpleNamespace(visual_input=state, active_ids=torch.tensor([0]), skip_ids=torch.tensor([1, 2]),
                           ghost_ids=torch.tensor([1]), predicted_residual=prediction,
                           react_logits=parameter.expand(1, 2), next_decision=1),
        1: SimpleNamespace(visual_input=later, active_ids=torch.tensor([1]), skip_ids=torch.tensor([0, 2]),
                           ghost_ids=torch.tensor([], dtype=torch.long), predicted_residual=prediction[:, :0],
                           react_logits=None, next_decision=None)}
    teacher = {index: TeacherLayer(torch.zeros_like(state), torch.zeros_like(state) if teacher_zero else torch.ones_like(state)) for index in trace}
    decisions = {0: SimpleNamespace(selected_mask=torch.tensor([[True, False, False]])),
                 1: SimpleNamespace(selected_mask=torch.tensor([[False, True, False]]))}
    return trace, decisions, teacher, frozen


def test_four_losses_future_labels_and_cross_frozen_layer_gradient():
    parameter = torch.nn.Parameter(torch.tensor(0.2))
    trace, decisions, teacher, frozen = trajectory(parameter)
    bundle = compute_losses(trace, decisions, teacher)
    assert bundle.ghost_count == 1 and bundle.candidate_count == 2 and bundle.reactivation_count == 1
    assert bundle.residual.item() == pytest.approx(0.64)
    assert bundle.freshness.item() == pytest.approx(0.16)
    assert bundle.reactivation.item() == pytest.approx(F.binary_cross_entropy_with_logits(torch.tensor([0.2, 0.2]), torch.tensor([1., 0.])).item())
    bundle.freshness.backward()
    assert parameter.grad.item() == pytest.approx(1.6)
    assert frozen.weight.grad is None


def test_warmup_no_fresh_zero_direction_and_empty_sets():
    parameter = torch.nn.Parameter(torch.tensor(0.2))
    trace, decisions, teacher, _ = trajectory(parameter, teacher_zero=True)
    bundle = compute_losses(trace, decisions, teacher, stage="warmup")
    assert bundle.freshness.item() == 0 and bundle.direction.item() == 0
    assert bundle.direction_count == 0
    assert compute_losses(trace, decisions, teacher, ablation="no_fresh").freshness.item() == 0
    assert compute_losses(trace, decisions, teacher, ablation="score").reactivation.item() == 0
    empty = compute_losses({}, {}, {}, zero_anchor=parameter)
    assert empty.total.item() == 0 and torch.isfinite(empty.total)
    empty.total.backward()
    assert parameter.grad.item() == 0


def test_missing_next_decision_is_error():
    parameter = torch.nn.Parameter(torch.tensor(0.2))
    trace, decisions, teacher, _ = trajectory(parameter)
    with pytest.raises(ValueError, match="future decision"):
        compute_losses(trace, {}, teacher)


def tiny_modules():
    torch.manual_seed(42)
    modules = GhostModules(4, 4, GhostConfig(bottleneck_dim=2, num_prototypes=2, share_every=1))
    metadata = {"backbone": "synthetic", "hidden_size": 4, "num_layers": 4,
                "drop_layers": [0, 2], "keep_ratios": [0.5, 0.5], "ablation": "full",
                "ghost_config": asdict(modules.config), "manifest_sha256": "b" * 64}
    return modules, metadata


def test_two_stage_training_and_resume_identical(tmp_path):
    def callback(modules):
        parameter = next(modules.parameters())
        def run(index, stage):
            p = parameter.mean() + index * 0.01
            trace, decisions, teacher, _ = trajectory(p)
            return compute_losses(trace, decisions, teacher, stage=stage)
        return run
    full, metadata = tiny_modules()
    trainer = TwoStageTrainer(full, metadata, tmp_path / "full.pt", gradient_accumulation=2)
    phases = []
    trainer.run(3, callback(full), on_step=lambda state, stage, rows: phases.append(stage))
    assert trainer.state["complete"] and phases == ["warmup", "warmup", "rollout", "rollout"]
    interrupted, _ = tiny_modules()
    first = TwoStageTrainer(interrupted, metadata, tmp_path / "resume.pt", gradient_accumulation=2)
    first.run(3, callback(interrupted), max_steps=1)
    assert not first.state["complete"]
    resumed, _ = tiny_modules()
    second = TwoStageTrainer(resumed, metadata, tmp_path / "resume.pt", gradient_accumulation=2)
    second.resume(tmp_path / "resume.pt")
    second.run(3, callback(resumed))
    assert second.state["complete"] and second.state["optimizer_steps"] == 4
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())


def test_optimizer_preserves_ablation_freezing_and_excludes_unused_tail():
    modules = GhostModules(4, 8, GhostConfig(ablation="self", share_every=1))
    optimizer = make_optimizer(modules, [0, 3])
    selected = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    for name, parameter in modules.named_parameters():
        if "prototype_queries" in name or "context_up" in name or name.startswith("blocks.7."):
            assert id(parameter) not in selected and not parameter.requires_grad
    no_decay = {id(parameter) for group in optimizer.param_groups if group["weight_decay"] == 0 for parameter in group["params"]}
    for name, parameter in modules.named_parameters():
        if parameter.requires_grad and ("gate" in name or "reactivation" in name):
            assert id(parameter) in no_decay

@pytest.mark.parametrize('factory_name,family', [('make_llava', 'llava'), ('make_qwen', 'qwen25vl')])
def test_real_tiny_backbone_two_stage_teacher_training(tmp_path, factory_name, family):
    import importlib.util
    from pathlib import Path
    from cold_ghost.patching import patch_model_with_cold_ghost
    from cold_ghost.training import forward_training_sample
    from models.router import PDropRouter
    path = Path(__file__).with_name('test_patching.py')
    spec = importlib.util.spec_from_file_location('tiny_cold_ghost_factories', path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    torch.manual_seed(42)
    model, sample = getattr(fixtures, factory_name)()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    modules = GhostModules(16, 5, GhostConfig(ghost_budget=1))
    metadata = {'backbone': 'synthetic-' + family, 'hidden_size': 16, 'num_layers': 5,
                'drop_layers': [1, 3], 'keep_ratios': [0.5, 0.5], 'ablation': 'full',
                'ghost_config': asdict(modules.config), 'manifest_sha256': 'c' * 64}
    ctx = patch_model_with_cold_ghost(model, PDropRouter(drop_layers=[1, 3], keep_ratios=[0.5, 0.5], monotonic=False),
                                     model_family=family, ghost_modules=modules, collect_trace=True,
                                     apply_updates=False, checkpoint_layers=True)
    trainer = TwoStageTrainer(modules, metadata, tmp_path / (family + '.pt'), gradient_accumulation=1, smoke=True)
    stages = []
    before = {name: parameter.detach().clone() for name, parameter in modules.named_parameters()}
    def forward(index, stage):
        losses, teacher = forward_training_sample(model, ctx, sample, stage=stage)
        assert len(teacher) == 5 and all(not layer.visual_input.requires_grad for layer in teacher.values())
        assert torch.isfinite(losses.total) and losses.total.requires_grad
        assert ctx.apply_updates == (stage == 'rollout')
        stages.append(stage)
        return losses
    trainer.run(1, forward)
    assert trainer.state['complete'] and stages == ['warmup', 'rollout']
    assert any(not torch.equal(before[name], parameter) for name, parameter in modules.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        assert torch.equal(original[name], parameter)

def test_diagnostic_ap_ties_and_no_positive_groups():
    from cold_ghost.diagnostics import average_precision, summarize_candidates
    assert average_precision([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert average_precision([0.9, 0.2], [1, 0]) == pytest.approx(1.0)
    assert average_precision([0.1, 0.2], [0, 0]) is None
    report = summarize_candidates([{'scores': [0.2], 'labels': [0], 'ghost_count': 1,
                                    'positive_count': 0, 'true_selected': 0,
                                    'errors': {'next_reactivated': [], 'next_not_reactivated': [{'mse': 1.0, 'cosine_error': 0.5}]}}])
    assert report['recall'] is None and report['average_precision'] is None
    assert report['next_reactivated']['mse'] is None
    assert report['next_not_reactivated']['count'] == 1
