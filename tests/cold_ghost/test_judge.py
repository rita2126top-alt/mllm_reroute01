import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cold_ghost.judge import (DEFAULT_API_URL, DEFAULT_MODEL, FAILURE_MARKER,
    FALLBACK_RATIONALE, JUDGE_MODULE, RETRY_MARKER, mmbench_failure_logging,
    scan_mmbench_judge_log, validate_mmbench_judge_environment)


class JudgeProtocolTests(unittest.TestCase):
    def test_environment_validation_returns_no_credential(self):
        secret = "dummy-credential-never-returned"
        result = validate_mmbench_judge_environment({"OPENAI_API_KEY": secret})
        self.assertEqual(result, {"api_type": "openai", "api_url": DEFAULT_API_URL,
                                  "model": DEFAULT_MODEL, "key_configured": True})
        self.assertNotIn(secret, json.dumps(result))

    def test_environment_rejects_missing_placeholders_and_changed_protocol(self):
        for value in ("", "YOUR_API_KEY", "<your-key>", "$OPENAI_API_KEY", "  key", "key with spaces"):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                validate_mmbench_judge_environment({"OPENAI_API_KEY": value})
            self.assertNotIn(repr(value), str(caught.exception))
        for name, value in (("API_TYPE", "azure"), ("OPENAI_API_URL", "https://other.invalid/secret"),
                            ("MODEL_VERSION", "changed-model")):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                validate_mmbench_judge_environment({"OPENAI_API_KEY": "dummy-key", name: value})
            self.assertNotIn(value, str(caught.exception))

    @staticmethod
    def fake_modules():
        from loguru import logger
        class Evaluator:
            def extract_answer_from_item(self, item):
                if item.get("retry"):
                    logger.patch(lambda record: record.update(name=JUDGE_MODULE)).info(
                        "GPT API failed to answer. ")
                if item.get("unparseable"):
                    logger.patch(lambda record: record.update(name=JUDGE_MODULE)).info(
                        'GPT output includes 0 / >1 letter in "ABCD": private-judge-answer')
                if item.get("fallback"):
                    return "E", FALLBACK_RATIONALE + " "
                return "A", "valid judge answer"
        module = types.SimpleNamespace(MMBench_Evaluator=Evaluator)
        def import_module(name):
            if name == "lmms_eval.models":
                return types.SimpleNamespace()
            if name == JUDGE_MODULE:
                return module
            raise AssertionError(name)
        return Evaluator, import_module

    def test_recovered_retries_preserve_original_return_and_do_not_fail(self):
        evaluator, importer = self.fake_modules()
        original = evaluator.extract_answer_from_item
        output = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True), \
             patch("cold_ghost.judge.importlib.import_module", side_effect=importer), \
             contextlib.redirect_stdout(output):
            with mmbench_failure_logging(["mmbench_en_dev"]):
                self.assertEqual(evaluator().extract_answer_from_item({"retry": True, "unparseable": True}),
                                 ("A", "valid judge answer"))
        self.assertIs(evaluator.extract_answer_from_item, original)
        self.assertIn(RETRY_MARKER, output.getvalue())
        self.assertNotIn(FAILURE_MARKER, output.getvalue())
        self.assertNotIn("private-judge-answer", output.getvalue())

    def test_actual_fallback_preserves_return_then_fails_on_context_exit(self):
        evaluator, importer = self.fake_modules()
        original = evaluator.extract_answer_from_item
        output = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True), \
             patch("cold_ghost.judge.importlib.import_module", side_effect=importer), \
             contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "random fallback"):
                with mmbench_failure_logging("mmbench_en_dev"):
                    result = evaluator().extract_answer_from_item({"fallback": True})
                    self.assertEqual(result, ("E", FALLBACK_RATIONALE + " "))
        self.assertIs(evaluator.extract_answer_from_item, original)
        self.assertEqual(output.getvalue().strip(), FAILURE_MARKER + " random_fallback")

    def test_other_tasks_need_no_judge_import_or_credential(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("cold_ghost.judge.importlib.import_module", side_effect=AssertionError("unexpected import")):
            with mmbench_failure_logging(["pope", "gqa"]):
                pass

    def test_log_scanner_rejects_only_terminal_marker_and_never_returns_log_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(RETRY_MARKER + " api_retry_group_exhausted\n"
                            + "model answer contains " + FAILURE_MARKER + " random_fallback\n"
                            + FAILURE_MARKER + " random_fallback\n"
                            + "private token not returned\n", encoding="utf-8")
            self.assertEqual(scan_mmbench_judge_log(path), [{"code": "random_fallback", "line": 3}])

    def test_direct_evaluation_retains_results_and_marks_actual_judge_failure(self):
        from scripts.cold_ghost import evaluate
        evaluator, importer = self.fake_modules()
        def fake_evaluate(*args, **kwargs):
            evaluator().extract_answer_from_item({"fallback": True})
            return {"results": {"mmbench_en_dev": {"gpt_eval_score,none": 0.0}}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True), \
                 patch("cold_ghost.judge.importlib.import_module", side_effect=importer), \
                 patch.object(evaluate, "run_evaluation", side_effect=fake_evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "random fallback"):
                    evaluate.main(["--config", "baseline/llava15", "--original",
                                   "--tasks", "mmbench_en_dev", "--out", str(output)])
            self.assertTrue((output / "results.json").is_file())
            self.assertEqual(json.loads((output / "failure.json").read_text())["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
