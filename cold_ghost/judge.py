"""Observe the unchanged MMBench judge without exposing credentials or changing scores.

lmms-eval 0.7.1 discards the rationale returned by its answer extractor. Its
INFO-level retry logs are also suppressed by the model registry's WARNING sink.
The scoped observer below preserves every extractor result, but records the exact
random-fallback return before it is discarded. Recovered retries remain valid.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-2024-11-20"
FAILURE_MARKER = "COLD_GHOST_MMBENCH_JUDGE_FAILURE"
RETRY_MARKER = "COLD_GHOST_MMBENCH_JUDGE_RETRY"
FALLBACK_RATIONALE = "Failed to predict, thus randomly generate one."
JUDGE_MODULE = "lmms_eval.tasks.mmbench.mmbench_evals"


def validate_mmbench_judge_environment(environ=None) -> dict:
    """Check the fixed release judge locally; do not call an API or return keys."""
    env = os.environ if environ is None else environ
    expected = {"API_TYPE": "openai", "OPENAI_API_URL": DEFAULT_API_URL,
                "MODEL_VERSION": DEFAULT_MODEL}
    for name, value in expected.items():
        if env.get(name, value) != value:
            raise ValueError(f"{name} must use the original MMBench judge default")
    key = env.get("OPENAI_API_KEY", "")
    normalized = key.strip().lower() if isinstance(key, str) else ""
    placeholders = {"your_api_key", "your-api-key", "your api key", "api_key", "api-key",
                    "changeme", "change_me", "replace_me", "none", "null", "todo", "placeholder"}
    if (not normalized or normalized in placeholders or key != key.strip()
            or any(char.isspace() for char in key) or "$" in key or "<" in key or ">" in key
            or normalized.startswith(("your_", "your-", "replace_", "replace-"))):
        raise ValueError("OPENAI_API_KEY must contain a configured credential, not a placeholder")
    return {"api_type": "openai", "api_url": DEFAULT_API_URL,
            "model": DEFAULT_MODEL, "key_configured": True}


def _uses_mmbench(tasks) -> bool:
    if isinstance(tasks, str):
        from .config import resolve_tasks
        tasks = resolve_tasks(tasks)
    return "mmbench_en_dev" in tasks


@contextmanager
def mmbench_failure_logging(tasks):
    """Retain original results, then fail if the judge actually returned random data.

    Include writing results.json INSIDE this context: a failed judge's original
    outputs then remain reviewable even though the evaluation exits nonzero.
    This must surround an isolated evaluation, as the temporary observer is global
    to the MMBench class. Formal runner jobs already use separate subprocesses.
    """
    if not _uses_mmbench(tasks):
        yield
        return
    validate_mmbench_judge_environment()
    # Import this before installing our sink: the registry removes existing sinks.
    importlib.import_module("lmms_eval.models")
    module = importlib.import_module(JUDGE_MODULE)
    from loguru import logger
    evaluator_class = module.MMBench_Evaluator
    original = evaluator_class.extract_answer_from_item
    failures = []

    def emit(marker, code):
        # Never emit original messages, exception text, predictions or credentials.
        # Start a fresh line even when a progress bar left a partial line.
        print(f"\n{marker} {code}", file=sys.stdout, flush=True)

    def observe_retry(message):
        text = message.record["message"]
        if text.startswith("GPT API failed to answer."):
            emit(RETRY_MARKER, "api_retry_group_exhausted")
        elif text.startswith('GPT output includes 0 / >1 letter in "ABCD":'):
            emit(RETRY_MARKER, "judge_answer_unparseable")

    def observe_result(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if (isinstance(result, (tuple, list)) and len(result) == 2
                and isinstance(result[1], str)
                and result[1].strip() == FALLBACK_RATIONALE):
            failures.append("random_fallback")
            emit(FAILURE_MARKER, "random_fallback")
        return result

    sink = logger.add(observe_retry, level="INFO", format="{message}", catch=False,
                      filter=lambda record: record["name"] == JUDGE_MODULE)
    try:
        with patch.object(evaluator_class, "extract_answer_from_item", observe_result):
            yield
    finally:
        logger.remove(sink)
    if failures:
        raise RuntimeError("MMBench judge used random fallback; retained results require review and are not a successful formal evaluation")


def scan_mmbench_judge_log(path: str | Path) -> list[dict]:
    """Return only safe failure codes/line numbers, never arbitrary log contents.

    Only the observer's anchored terminal-failure marker is authoritative.
    INFO retry messages and unparseable intermediate answers may recover, so they
    do not prove random fallback. Original 0.7.1 drops fallback rationales before
    creating results/samples; searching model predictions would cause false hits.
    """
    pattern = re.compile(r"^" + re.escape(FAILURE_MARKER) + r" (random_fallback)\s*$")
    findings = []
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.fullmatch(line.rstrip("\r\n"))
            if match:
                findings.append({"code": match.group(1), "line": line_number})
    return findings
