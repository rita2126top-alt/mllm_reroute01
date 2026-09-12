#!/usr/bin/env python
"""Measure all three proposed diagnostics on the independent validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True, help="Completed full-method checkpoint")
    parser.add_argument("--self-checkpoint", required=True, help="Separately trained Self-only checkpoint, same manifest/schedule")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from cold_ghost.checkpoint import load_checkpoint
    from cold_ghost.config import load_experiment
    from cold_ghost.data import PromptDataset, load_manifest
    from cold_ghost.diagnostics import (candidate_diagnostics, grouped_events, reactivation_events,
                                        residual_rank_diagnostics, summarize_candidates, summarize_rank)
    from cold_ghost.ghost import GhostConfig, GhostModules
    from cold_ghost.loading import build_inputs, gqa_prompt, load_backbone, make_router, validate_checkpoint
    from cold_ghost.patching import patch_model_with_cold_ghost
    from cold_ghost.training import compute_losses, dense_teacher_capture, freeze_backbone, mean_reports

    spec = load_experiment(args.config)
    manifest = load_manifest(args.manifest)
    validation = PromptDataset(manifest, args.images, "validation")
    full_payload = validate_checkpoint(spec, args.checkpoint, "full")
    self_payload = validate_checkpoint(spec, args.self_checkpoint, "self")
    for payload in (full_payload, self_payload):
        if payload["metadata"]["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Diagnostic checkpoint and validation manifest differ")
    model, processor = load_backbone(spec, device=args.device)
    freeze_backbone(model)
    modules = {}
    for name, path, payload in (("full", args.checkpoint, full_payload), ("self", args.self_checkpoint, self_payload)):
        metadata = payload["metadata"]
        modules[name] = GhostModules(metadata["hidden_size"], metadata["num_layers"], GhostConfig(**metadata["ghost_config"])).to(args.device).eval()
        load_checkpoint(path, modules[name], expected_metadata={"manifest_sha256": manifest["manifest_sha256"]})
    ctx = patch_model_with_cold_ghost(model, make_router(spec), action="compact_route", model_family=spec.model_family,
                                     ghost_modules=modules["full"], collect_trace=True, detach_trace=True)
    events = {"reroute": [], "full": [], "paired_reroute": [], "paired_full": []}
    candidate_reports, rank_rows, losses = [], [], {"full": [], "self": []}
    with torch.no_grad():
        for index in range(len(validation)):
            sample = validation[index]
            inputs = build_inputs(processor, model, sample["image"], gqa_prompt(sample["question"]))
            kwargs = dict(inputs, use_cache=False, return_dict=True)
            with dense_teacher_capture(ctx) as teacher:
                output = model(**kwargs)
                del output
            ctx.apply_updates = False
            ctx.ghost_modules = modules["full"]
            output = model(**kwargs)
            del output
            original_events = reactivation_events(ctx.trace, ctx.decisions, teacher)
            events["reroute"].extend(original_events.values())
            ctx.apply_updates = True
            output = model(**kwargs)
            del output
            gc_events = reactivation_events(ctx.trace, ctx.decisions, teacher)
            events["full"].extend(gc_events.values())
            for key in original_events.keys() & gc_events.keys():
                events["paired_reroute"].append(original_events[key])
                events["paired_full"].append({**gc_events[key], "age": original_events[key]["age"]})
            losses["full"].append(compute_losses(ctx.trace, ctx.decisions, teacher).report())
            candidate_reports.append(candidate_diagnostics(ctx.trace, ctx.decisions, teacher))
            rank_rows.extend(residual_rank_diagnostics(ctx.trace, teacher))
            ctx.ghost_modules = modules["self"]
            output = model(**kwargs)
            del output
            losses["self"].append(compute_losses(ctx.trace, ctx.decisions, teacher, ablation="self").report())
            print(f"Internal validation diagnostics: {index + 1}/{len(validation)}", flush=True)
    result = {"config": spec.config_name, "manifest_sha256": manifest["manifest_sha256"],
               "samples": len(validation), "staleness_by_full_skip_age": {name: grouped_events(rows) for name, rows in events.items()},
               "paired_event_count": len(events["paired_full"]),
               "dense_residual_rank_reconstruction": summarize_rank(rank_rows),
               "deployed_residual_prediction": {name: mean_reports(rows) for name, rows in losses.items()},
               "future_reactivation": summarize_candidates(candidate_reports),
               "note": "Self and full use their actual independently trained trajectories; paired staleness includes only common events and groups both paths by original Reroute skip age."}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
