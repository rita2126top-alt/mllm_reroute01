import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from cold_ghost.evaluation import MatrixJob, build_matrix, new_output_dir, release_eval_environment, run_matrix

class EvaluationIsolationTests(unittest.TestCase):
    def test_matrix_reuses_checkpoint_for_both_dispatch_forms(self):
        jobs = build_matrix(checkpoint_dir="weights")
        self.assertEqual(len(jobs), 62)
        gc_jobs = [job for job in jobs if job.arm == "gc"]
        self.assertEqual(len(gc_jobs), 24)
        self.assertEqual(len({job.checkpoint for job in gc_jobs}), 12)
        self.assertEqual(len({job.key for job in jobs}), 62)

    def test_gc_matrix_requires_checkpoint_root(self):
        with self.assertRaises(ValueError):
            build_matrix()
        self.assertEqual(len(build_matrix(arms="original")), 38)

    def test_dry_run_does_not_create_outputs_or_launch_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "uncreated"
            jobs = build_matrix(models=("llava15",), tiers=("avg64",), arms="original")
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = run_matrix(jobs, path, dry_run=True,
                                  runner=lambda *a, **k: self.fail("must not launch"))
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(stream.getvalue())), 7)
            self.assertFalse(path.exists())

    def test_process_failure_is_reported_and_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results"
            job = MatrixJob("baseline/llava15", "original", None, "full", "pope")
            code = run_matrix([job], path, runner=lambda *a, **k: subprocess.CompletedProcess(a[0], 9))
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["runs"][0]["exit_code"], 9)

    def test_missing_results_is_a_failure_even_on_zero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = MatrixJob("baseline/llava15", "original", None, "full", "pope")
            self.assertEqual(run_matrix([job], Path(temporary) / "results",
                runner=lambda *a, **k: subprocess.CompletedProcess(a[0], 0)), 1)

    def test_output_directory_cannot_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                new_output_dir(temporary)

    def test_grounding_preset_environment_restored(self):
        with patch.dict(os.environ, {"REFCOCO_PROMPT_STYLE": "custom"}, clear=True):
            with release_eval_environment():
                os.environ["REFCOCO_PROMPT_STYLE"] = "full"
                os.environ["BBOX_COORD_FORMAT"] = "normalized"
            self.assertEqual(os.environ["REFCOCO_PROMPT_STYLE"], "custom")
            self.assertNotIn("BBOX_COORD_FORMAT", os.environ)

class RegisteredFlopTests(unittest.TestCase):
    def test_linear_explicit_score_and_fused_sdpa_are_all_counted(self):
        import torch
        from cold_ghost.profiling import registered_flop_counter
        layer = torch.nn.Linear(4, 5, bias=False)
        value = torch.randn(2, 3, 4)
        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 2, 5, 4)
        attention_value = torch.randn(1, 2, 5, 4)
        with torch.inference_mode(), registered_flop_counter() as counter:
            layer(value)
            query @ key.transpose(-2, -1)
            torch.nn.functional.scaled_dot_product_attention(query, key, attention_value)
        # Linear 2*6*4*5=240, scoring QK=240, SDPA QK+AV=480.
        self.assertEqual(counter.get_total_flops(), 960)
        counts = counter.get_flop_counts()["Global"]
        self.assertTrue(any("attention" in str(op) or "bmm" in str(op) for op in counts))

    def test_gqa_prompt_matches_v071_default(self):
        from cold_ghost.loading import gqa_prompt
        self.assertEqual(gqa_prompt("What color?"),
                         "What color?\nAnswer the question using a single word or phrase.")

class CheckpointInferenceTests(unittest.TestCase):
    def test_formal_checkpoint_rejects_smoke_and_changed_hyperparameters(self):
        import copy
        from dataclasses import asdict
        from cold_ghost.config import load_experiment
        from cold_ghost.ghost import GhostConfig
        from cold_ghost.loading import validate_checkpoint
        spec = load_experiment("llava15/avg192/llava15_ours_vs_fastv_avg192")
        payload = {"metadata": {**spec.checkpoint_metadata(), "hidden_size": 4096,
                   "ablation": "full", "ghost_config": asdict(GhostConfig()),
                   "manifest_sha256": "a" * 64},
                   "training_state": {"complete": True, "smoke": False}}
        with patch("cold_ghost.checkpoint.inspect_checkpoint", return_value=payload):
            self.assertIs(validate_checkpoint(spec, "unused.pt"), payload)
        for change in ("smoke", "budget", "layers", "incomplete"):
            bad = copy.deepcopy(payload)
            if change == "smoke":
                bad["training_state"]["smoke"] = True
            elif change == "budget":
                bad["metadata"]["ghost_config"]["ghost_budget"] = 1
            elif change == "layers":
                bad["metadata"]["drop_layers"] = [3]
            else:
                bad["training_state"]["complete"] = False
            with patch("cold_ghost.checkpoint.inspect_checkpoint", return_value=bad):
                with self.assertRaises(ValueError):
                    validate_checkpoint(spec, "unused.pt")

    def test_efficiency_matrix_dispatches_correct_entry_point(self):
        jobs = build_matrix(models=("llava15",), tiers=("avg192",), arms="original")
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                run_matrix(jobs, Path(temporary) / "new", dry_run=True,
                           mode="benchmark", n_passes=5, n_warmup_passes=2, n_decode_tokens=64)
            plan = json.loads(stream.getvalue())
            self.assertTrue(plan[0]["command"][1].endswith("benchmark.py"))
            self.assertIn("--n-decode-tokens", plan[0]["command"])
            self.assertNotIn("--tasks", plan[0]["command"])

class ReleaseAdapterIntegrationTests(unittest.TestCase):
    def test_real_release_eval_function_keeps_model_and_task_defaults(self):
        import sys
        import types
        from cold_ghost.evaluation import run_evaluation
        calls = {}
        wrapper = types.SimpleNamespace(_model=object())
        class FakeModelClass:
            @staticmethod
            def create_from_arg_string(arguments, additional):
                calls["arguments"] = arguments
                calls["additional"] = additional
                return wrapper
        package = types.ModuleType("lmms_eval")
        models = types.ModuleType("lmms_eval.models")
        def evaluate(**kwargs):
            calls["evaluation"] = kwargs
            calls["preset"] = (os.environ.get("BBOX_COORD_FORMAT"),
                               os.environ.get("REFCOCO_PROMPT_STYLE"))
            return {"results": {"pope": {"accuracy": 1.0}}}
        package.evaluator = types.SimpleNamespace(simple_evaluate=evaluate)
        models.get_model = lambda name: FakeModelClass
        with patch.dict(sys.modules, {"lmms_eval": package, "lmms_eval.models": models}):
            with patch.dict(os.environ, {}, clear=True):
                result = run_evaluation("baseline/qwen25vl", tasks="paper_main,vqa", limit=2)
                self.assertNotIn("BBOX_COORD_FORMAT", os.environ)
        self.assertEqual(calls["preset"], ("pixel", "nuwa"))
        self.assertIn("max_pixels=451584", calls["arguments"])
        self.assertIn("attn_implementation=sdpa", calls["arguments"])
        self.assertEqual(calls["additional"], {"batch_size": 1})
        self.assertEqual(len(calls["evaluation"]["tasks"]), 12)
        self.assertEqual(calls["evaluation"]["limit"], 2)
        self.assertNotIn("gen_kwargs", calls["evaluation"])
        self.assertIn("results", result)

    def test_gc_replaces_only_release_routing_installation(self):
        import sys
        import types
        from cold_ghost.evaluation import run_evaluation
        wrapper = types.SimpleNamespace(_model=object())
        class FakeModelClass:
            @staticmethod
            def create_from_arg_string(arguments, additional):
                return wrapper
        package = types.ModuleType("lmms_eval")
        models = types.ModuleType("lmms_eval.models")
        package.evaluator = types.SimpleNamespace(simple_evaluate=lambda **kw: {"results": {"pope": {"accuracy": 1.0}}})
        models.get_model = lambda name: FakeModelClass
        marker = object()
        with patch.dict(sys.modules, {"lmms_eval": package, "lmms_eval.models": models}):
            with patch("cold_ghost.loading.validate_checkpoint"), patch("cold_ghost.loading.patch_experiment", return_value=marker) as install:
                run_evaluation("llava15/avg192/llava15_ours_vs_fastv_avg192",
                               checkpoint="trained.pt", tasks="pope")
        self.assertIs(wrapper._routing_ctx, marker)
        self.assertEqual(install.call_count, 1)
        self.assertIs(install.call_args.args[0], wrapper._model)
        self.assertEqual(install.call_args.args[1].routing["drop_layers"], [3, 7, 15, 23])

if __name__ == "__main__":
    unittest.main()
