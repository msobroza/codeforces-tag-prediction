"""Tests for running the LLM judges."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from codeforces_tags.config import (
    AUDIT_JUDGE_EFFORT,
    AUDIT_JUDGE_MODEL,
    NEAR_DUPLICATE_AUDIT_DIR,
)
from codeforces_tags.near_duplicates.audit import judges
from codeforces_tags.near_duplicates.audit.judges import (
    JudgeSettings,
    check_labels,
    judge,
    judge_all,
    sdk_version,
)

SCHEMA = {"type": "object"}


def result_query(*results):
    """Return a query that streams the given messages, whatever the prompt."""

    async def query(prompt, options):  # pylint: disable=unused-argument
        for message in results:
            yield message

    return query


def result(labels=None, is_error=False):
    """Build a result message like the SDK's ResultMessage."""
    output = None if labels is None else {"labels": labels}
    return SimpleNamespace(
        structured_output=output, is_error=is_error, total_cost_usd=0.5, usage={"x": 1}
    )


def test_settings_default_to_the_recorded_run_and_an_agent_confined_to_its_folder():
    options = JudgeSettings().options(SCHEMA, Path("/work/texts"))

    assert (AUDIT_JUDGE_MODEL, AUDIT_JUDGE_EFFORT) == ("claude-opus-5-5", "xhigh")
    assert options == {
        "model": "claude-opus-5-5",
        "effort": "xhigh",
        "tools": ["Read"],  # the only tool the model sees
        "allowed_tools": ["Read(./**)"],  # and only inside its folder
        "permission_mode": "dontAsk",  # anything else is refused
        "setting_sources": [],  # no CLAUDE.md
        "cwd": "/work/texts",
        "output_format": {"type": "json_schema", "schema": SCHEMA},
    }


def test_judge_returns_the_answer_with_its_cost_harness_and_tool_calls():
    read = SimpleNamespace(name="Read", input={"file_path": "batch_0.json"})
    query = result_query(
        SimpleNamespace(subtype="init", data={"claude_code_version": "2.1.286"}),
        SimpleNamespace(content=[SimpleNamespace(text="thinking"), read]),
        SimpleNamespace(content="a plain user message"),
        result([{"pair_id": 1}]),
    )

    run = asyncio.run(judge("prompt", SCHEMA, JudgeSettings(), Path("."), query))

    assert run.labels == [{"pair_id": 1}]
    assert (run.cost_usd, run.usage) == (0.5, {"x": 1})
    assert run.claude_code_version == "2.1.286"
    assert run.tool_calls == [{"name": "Read", "input": {"file_path": "batch_0.json"}}]


@pytest.mark.parametrize(
    "messages",
    [(), (result(labels=None),), (result([{"pair_id": 1}], is_error=True),)],
    ids=["no result", "no structured output", "error"],
)
def test_judge_fails_without_a_valid_answer(messages):
    with pytest.raises(RuntimeError, match="no structured answer"):
        asyncio.run(judge("prompt", SCHEMA, JudgeSettings(), Path("."), result_query(*messages)))


def test_judge_all_reports_a_failed_judge_without_stopping_the_others():
    async def query(prompt, options):  # pylint: disable=unused-argument
        if prompt == "bad":
            raise RuntimeError("no answer")
        yield result([{"pair_id": 1}])

    good, bad = judge_all(["good", "bad"], SCHEMA, JudgeSettings(), Path("."), query)

    assert good.labels == [{"pair_id": 1}]
    assert isinstance(bad, RuntimeError)


def test_sdk_version_is_none_when_the_sdk_is_missing(monkeypatch):
    def missing(name):
        raise judges.PackageNotFoundError(name)

    monkeypatch.setattr(judges, "version", missing)
    assert sdk_version() is None
    monkeypatch.setattr(judges, "version", lambda name: "0.2.163")
    assert sdk_version() == "0.2.163"


def test_judge_all_keeps_the_order_of_the_prompts(fake_query, tmp_path):
    batch = tmp_path / "batch_0.json"
    batch.write_text('[{"pair_id": 4}]', encoding="utf-8")
    prompts = [f"Read the JSON file {batch} (", "Label ONLY these pair_ids: [9,2]."]

    runs = judge_all(prompts, SCHEMA, JudgeSettings(), tmp_path, fake_query)

    assert [[label["pair_id"] for label in run.labels] for run in runs] == [[4], [9, 2]]


def test_judge_uses_the_agent_sdk_by_default(monkeypatch):
    sdk = ModuleType("claude_agent_sdk")
    received = {}

    def options_class(**options):
        received.update(options)
        return options

    async def query(prompt, options):  # pylint: disable=unused-argument
        yield result([{"pair_id": 1}])

    sdk.ClaudeAgentOptions, sdk.query = options_class, query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)

    run = asyncio.run(judge("prompt", SCHEMA, JudgeSettings(), Path(".")))

    assert run.labels == [{"pair_id": 1}]
    assert received["effort"] == "xhigh"


def test_check_labels_accepts_each_pair_once_in_any_order():
    check_labels([{"pair_id": 2}, {"pair_id": 1}], [1, 2])


@pytest.mark.parametrize("given", [[1], [1, 2, 2], [1, 2, 3]], ids=["missing", "repeated", "extra"])
def test_check_labels_rejects_answers_that_do_not_match_the_batch(given):
    with pytest.raises(ValueError, match="do not match the batch"):
        check_labels([{"pair_id": pair_id} for pair_id in given], [1, 2])


def test_settings_match_the_recorded_run():
    metadata = json.loads((NEAR_DUPLICATE_AUDIT_DIR / "audit_metadata.json").read_text())

    assert metadata["judges"]["model"] == JudgeSettings().model
    assert metadata["judges"]["reasoning_effort"] == JudgeSettings().effort
