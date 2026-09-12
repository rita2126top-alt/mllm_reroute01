"""CPU integration tests use real Transformers decoders, no model downloads."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cold_ghost.ghost import GhostConfig, GhostModules
from cold_ghost.patching import patch_model_with_cold_ghost, unpatch_model_with_cold_ghost
from models.router import PDropRouter, FastVRouter
from models.dispatcher import TokenDispatcher
from models.patching import patch_model_for_routing


def make_llava():
    from transformers import LlavaConfig, LlavaForConditionalGeneration, LlamaConfig, CLIPVisionConfig
    text = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=2,
                       max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                              num_attention_heads=2, image_size=8, patch_size=4)
    config = LlavaConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         image_token_index=50, image_seq_length=4)
    config._attn_implementation = "sdpa"
    model = LlavaForConditionalGeneration(config).eval()
    model.requires_grad_(False)
    return model, dict(input_ids=torch.tensor([[1, 50, 50, 50, 50, 7, 9]]),
                       pixel_values=torch.randn(1, 3, 8, 8), attention_mask=torch.ones(1, 7, dtype=torch.long))


def make_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=128,
                         rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
                         bos_token_id=1, eos_token_id=2, pad_token_id=0),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32,
                           num_heads=2, in_channels=3, patch_size=2,
                           spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=16, fullatt_block_indexes=[0, 1], window_size=8),
        image_token_id=50, video_token_id=51, vision_start_token_id=52, vision_end_token_id=53)
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    model.requires_grad_(False)
    return model, dict(input_ids=torch.tensor([[1, 52, 50, 50, 50, 50, 53, 7, 9]]),
                       pixel_values=torch.randn(16, 24), image_grid_thw=torch.tensor([[1, 4, 4]]),
                       attention_mask=torch.ones(1, 9, dtype=torch.long),
                       mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]]))


def router():
    return PDropRouter(drop_layers=[1, 3], keep_ratios=[0.5, 0.5], monotonic=False)


def modules():
    return GhostModules(16, 5, GhostConfig(ghost_budget=1))


@pytest.mark.parametrize("factory,family", [(make_llava, "llava"), (make_qwen, "qwen25vl")])
def test_real_prefill_stagewise_matches_compact_and_updates_deferred(factory, family):
    torch.manual_seed(7)
    first, sample = factory()
    second = copy.deepcopy(first)
    ghost = modules()
    a = patch_model_with_cold_ghost(first, router(), ghost_modules=ghost,
                                    model_family=family, collect_trace=True)
    b = patch_model_with_cold_ghost(second, router(), action="compact_route_stagewise",
                                    ghost_modules=copy.deepcopy(ghost), model_family=family, collect_trace=True)
    with torch.no_grad():
        out_a = first(**sample, use_cache=False)
        out_b = second(**sample, use_cache=False)
    torch.testing.assert_close(out_a.logits[:, -1], out_b.logits[:, -1], rtol=1e-5, atol=1e-6)
    for layer in range(5):
        torch.testing.assert_close(a.trace[layer].visual_input, b.trace[layer].visual_input)
        assert torch.equal(a.trace[layer].active_ids, b.trace[layer].active_ids)
        assert torch.equal(a.trace[layer].ghost_ids, b.trace[layer].ghost_ids)
    assert a.counters["ghost_token_layers"] == 2
    assert a.trace[3].ghost_ids.numel() == 0
    assert a.trace[4].react_logits is None
    assert a.counters == b.counters
    assert b.auxiliary_state_bytes() > 0
    for layer in (1, 2):
        trace = a.trace[layer]
        cold = trace.skip_ids[~torch.isin(trace.skip_ids, trace.ghost_ids)]
        torch.testing.assert_close(a.trace[layer + 1].visual_input[:, cold], trace.visual_input[:, cold])
        torch.testing.assert_close(a.trace[layer + 1].visual_input[:, trace.ghost_ids],
                                   trace.visual_input[:, trace.ghost_ids] + trace.predicted_residual)
    unpatch_model_with_cold_ghost(first)
    assert not hasattr(first, "_cold_ghost_context")


@pytest.mark.parametrize("factory,family", [(make_llava, "llava"), (make_qwen, "qwen25vl")])
@pytest.mark.parametrize("action", ["compact_route", "compact_route_stagewise"])
def test_real_two_decode_steps_preserve_compact_kv_and_reset(factory, family, action):
    model, sample = factory()
    ctx = patch_model_with_cold_ghost(model, router(), action=action, model_family=family, ghost_modules=modules())
    with torch.no_grad():
        first = model(**sample, use_cache=True)
        cache = first.past_key_values
        original_length = sample["input_ids"].shape[-1]
        assert cache.get_seq_length(0) == original_length
        assert cache.get_seq_length(1) == original_length - 2
        before = dict(ctx.counters)
        token = first.logits[:, -1].argmax(-1, keepdim=True)
        for step in range(2):
            position_ids = torch.tensor([[original_length + step]])
            if family == "qwen25vl":
                position_ids = position_ids.reshape(1, 1, 1).expand(3, 1, 1) + model.model.rope_deltas.reshape(1, 1, 1)
            output = model(input_ids=token, past_key_values=cache, use_cache=True, position_ids=position_ids,
                           attention_mask=torch.ones(1, original_length + step + 1, dtype=torch.long))
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            assert cache.get_seq_length(1) == original_length - 2 + step + 1
            assert torch.isfinite(output.logits).all()
        assert ctx.counters == before
        model(**sample, use_cache=False)
        assert ctx.counters == before


def test_zero_updates_match_original_router_and_full_outputs():
    torch.manual_seed(12)
    model, sample = make_llava()
    baseline = copy.deepcopy(model)
    ghost = modules()
    for block in ghost.blocks:
        nn.init.zeros_(block.up.weight)
        nn.init.zeros_(block.context_up.weight)
    ctx = patch_model_with_cold_ghost(model, router(), ghost_modules=ghost, collect_trace=True)
    # Dispatcher is not used by the original compact dispatch implementation.
    base_ctx = patch_model_for_routing(baseline, router(), None, "compact_route", "llava")
    with torch.no_grad():
        changed = model(**sample, use_cache=False)
        unchanged = baseline(**sample, use_cache=False)
    torch.testing.assert_close(changed.logits, unchanged.logits, rtol=0, atol=0)
    for layer, decision in ctx.decisions.items():
        assert torch.equal(decision.selected_mask, base_ctx.routing_log[layer].selected_mask)


def test_warmup_predictions_do_not_change_trajectory_and_rollout_gradients_cross_layers():
    torch.manual_seed(2)
    model, sample = make_llava()
    ghost = modules()
    ctx = patch_model_with_cold_ghost(model, router(), ghost_modules=ghost,
                                      collect_trace=True, apply_updates=False, checkpoint_layers=True)
    model(**sample, use_cache=False)
    assert ctx.trace[1].predicted_residual.requires_grad
    skip = ctx.trace[1].skip_ids
    torch.testing.assert_close(ctx.trace[2].visual_input[:, skip], ctx.trace[1].visual_input[:, skip])
    ctx.apply_updates = True
    model(**sample, use_cache=False)
    ctx.trace[1].predicted_residual.retain_grad()
    # Supervise the actual state before the next decision, including Ghost
    # corrections accumulated during the skipped interval.
    loss = ctx.trace[3].visual_input.square().mean()
    loss.backward()
    assert ctx.trace[1].predicted_residual.grad is not None
    assert ctx.trace[1].predicted_residual.grad.abs().sum() > 0
    assert ghost.blocks[0].up.weight.grad is not None


def test_empty_skip_single_decision_and_trace_detach():
    model, sample = make_llava()
    ctx = patch_model_with_cold_ghost(model, PDropRouter([1, 3], [1.0, 1.0], False),
                                      ghost_modules=modules(), collect_trace=True, detach_trace=True)
    model(**sample, use_cache=False)
    assert ctx.counters["ghost_token_layers"] == 0
    assert not ctx.trace[3].visual_input.requires_grad
    unpatch_model_with_cold_ghost(model)
    ctx = patch_model_with_cold_ghost(model, FastVRouter(scoring_layer=1, keep_ratio=0.5), ghost_modules=modules())
    model(**sample, use_cache=False)
    assert ctx.counters["estimated_macs"] == 0


def test_batch_and_padding_rejected():
    model, sample = make_llava()
    patch_model_with_cold_ghost(model, router(), ghost_modules=modules())
    sample["attention_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="unpadded"):
        model(**sample, use_cache=False)


@pytest.mark.parametrize("factory,family", [(make_llava, "llava"), (make_qwen, "qwen25vl")])
@pytest.mark.parametrize("action", ["compact_route", "compact_route_stagewise"])
def test_generate_real_hf_prefill_and_three_decode_tokens(factory, family, action):
    model, sample = factory()
    ctx = patch_model_with_cold_ghost(model, router(), action=action, model_family=family, ghost_modules=modules())
    with torch.no_grad():
        output = model.generate(**sample, min_new_tokens=3, max_new_tokens=3,
                                do_sample=False, use_cache=True, pad_token_id=0)
    assert output.shape[-1] == sample["input_ids"].shape[-1] + 3
    assert ctx.counters["ghost_token_layers"] == 2


def test_qwen_variable_visual_count_resets_stage_buffers():
    model, sample = make_qwen()
    ctx = patch_model_with_cold_ghost(model, router(), action="compact_route_stagewise",
                                     model_family="qwen25vl", ghost_modules=modules(), collect_trace=True)
    with torch.no_grad():
        model(**sample, use_cache=False)
        assert ctx.age.numel() == 4
        larger = dict(input_ids=torch.tensor([[1, 52] + [50] * 9 + [53, 7, 9]]),
                      pixel_values=torch.randn(36, 24), image_grid_thw=torch.tensor([[1, 6, 6]]),
                      attention_mask=torch.ones(1, 14, dtype=torch.long),
                      mm_token_type_ids=torch.tensor([[0, 0] + [1] * 9 + [0, 0, 0]]))
        model(**larger, use_cache=False)
    assert ctx.age.numel() == 9
    assert ctx.visual_token_range == (2, 11)
    assert ctx.trace[0].age.eq(0).all()
    assert ctx.decisions[1].selected_mask.shape == (1, 9)
    assert ctx.counters["ghost_token_layers"] == 2


def test_position_gather_preserves_qwen_three_dimensional_ids_and_mrope():
    from cold_ghost.patching import _gather_positions
    position_ids = torch.arange(21).reshape(3, 1, 7)
    cos = torch.randn(3, 1, 7, 8)
    sin = torch.randn(3, 1, 7, 8)
    kept = torch.tensor([0, 2, 5, 6])
    selected_ids, selected_pe = _gather_positions(position_ids, (cos, sin), kept)
    assert selected_ids.shape == (3, 1, 4)
    assert selected_pe[0].shape == (3, 1, 4, 8)
    assert torch.equal(selected_ids, position_ids[:, :, kept])
    assert torch.equal(selected_pe[0], cos[:, :, kept])
