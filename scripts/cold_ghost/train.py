#!/usr/bin/env python
"""Train one release backbone/schedule/tier Ghost checkpoint in two stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cold_ghost.bootstrap import isolate_script_directory
isolate_script_directory(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original compact or stagewise Reroute experiment config")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cold_ghost")
    parser.add_argument("--out", help="Explicit checkpoint filename")
    parser.add_argument("--ablation", choices=("full", "self", "context", "score", "no_fresh"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Stop after this total optimizer-step count; checkpoint remains partial")
    parser.add_argument("--smoke", action="store_true", help="Mark output ineligible for formal evaluation")
    parser.add_argument("--steps-per-stage", type=int, help="Shorten each stage; requires --smoke")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-activation-checkpointing", action="store_true", help="Debug only: store backbone activations")
    args = parser.parse_args()

    import torch
    from cold_ghost.checkpoint import experiment_metadata
    from cold_ghost.config import load_experiment
    from cold_ghost.data import PromptDataset, load_manifest
    from cold_ghost.ghost import GhostConfig, GhostModules
    from cold_ghost.loading import build_inputs, gqa_prompt, load_backbone, make_router
    from cold_ghost.patching import patch_model_with_cold_ghost
    from cold_ghost.training import TwoStageTrainer, forward_training_sample, freeze_backbone, mean_reports

    if args.steps_per_stage is not None and not args.smoke:
        parser.error("--steps-per-stage requires --smoke")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run CPU unit tests locally; real backbone training needs the server GPU.")
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    spec = load_experiment(args.config)
    if not spec.is_reroute:
        parser.error("Cold/Ghost training requires an original non-monotonic compact/stagewise Reroute config")
    manifest = load_manifest(args.manifest, allow_smoke=args.smoke)
    train = PromptDataset(manifest, args.images, "train")
    validation = PromptDataset(manifest, args.images, "validation")
    model, processor = load_backbone(spec, device=args.device)
    freeze_backbone(model)
    layers = model.model.language_model.layers
    hidden_size = layers[0].self_attn.q_proj.in_features
    modules = GhostModules(hidden_size, len(layers), GhostConfig(ablation=args.ablation)).to(args.device)
    metadata = experiment_metadata(spec, modules, manifest["manifest_sha256"])
    # Execution form is always compact during training; checkpoint identity ignores stagewise.
    ctx = patch_model_with_cold_ghost(model, make_router(spec), action="compact_route",
                                     model_family=spec.model_family, ghost_modules=modules,
                                     collect_trace=True, apply_updates=False,
                                     checkpoint_layers=not args.no_activation_checkpointing)
    output = Path(args.out) if args.out else Path(args.output_dir) / f"{spec.checkpoint_key}__{args.ablation}.pt"
    if output.exists() and not args.resume:
        parser.error(f"Checkpoint already exists: {output}. Resume explicitly or choose a new output.")
    trainer = TwoStageTrainer(modules, metadata, output, smoke=args.smoke,
                              steps_per_stage=args.steps_per_stage, save_every=args.save_every)
    if args.resume:
        trainer.resume(args.resume)
    log_path = output.with_suffix(".train.jsonl")

    def run_sample(dataset, index, stage):
        sample = dataset[index]
        question = gqa_prompt(sample["question"])
        inputs = build_inputs(processor, model, sample["image"], question)
        losses, _ = forward_training_sample(model, ctx, inputs, stage=stage, ablation=args.ablation)
        return losses

    def on_step(state, stage, reports):
        report = {"stage": stage, "optimizer_steps": state["optimizer_steps"],
                  "sample_cursor": state["sample_cursor"], **mean_reports(reports)}
        output.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)

    def validate(stage):
        modules.eval()
        reports = []
        try:
            with torch.no_grad():
                for index in range(len(validation)):
                    reports.append(run_sample(validation, index, stage).report())
        finally:
            modules.train()
        report = mean_reports(reports)
        print(json.dumps({"validation": stage, **report}), flush=True)
        return report

    result = trainer.run(len(train), lambda index, stage: run_sample(train, index, stage),
                         max_steps=args.max_steps, on_step=on_step, validate=validate)
    print(json.dumps({"checkpoint": str(output), "complete": result["complete"],
                      "smoke": result["smoke"], "optimizer_steps": result["optimizer_steps"],
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
