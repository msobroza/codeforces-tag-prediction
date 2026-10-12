"""Run LLM judges with the Claude Agent SDK: one agent per prompt, structured answers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from codeforces_tags.config import AUDIT_JUDGE_EFFORT, AUDIT_JUDGE_MODEL, AUDIT_JUDGE_TOOLS

# Sends a prompt with agent options and streams the agent's messages back.
Query = Callable[[str, dict[str, Any]], AsyncIterator[Any]]


@dataclass(frozen=True)
class JudgeSettings:
    """How a judge runs: model, reasoning effort and the only tools it may use."""

    model: str = AUDIT_JUDGE_MODEL
    effort: str = AUDIT_JUDGE_EFFORT
    tools: tuple[str, ...] = AUDIT_JUDGE_TOOLS

    def options(self, schema: dict, cwd: Path) -> dict[str, Any]:
        """Build the agent options.

        The agent sees only ``tools``, each allowed inside ``cwd`` alone: every other call is
        refused, so a judge cannot read anything but the files placed in its folder.

        Args:
            schema: JSON schema the answer must follow.
            cwd: Working directory of the agent, holding only what it may read.

        Returns:
            Keyword arguments for ``claude_agent_sdk.ClaudeAgentOptions``.
        """
        return {
            "model": self.model,
            "effort": self.effort,
            "tools": list(self.tools),  # the only tools the model sees
            "allowed_tools": [f"{tool}(./**)" for tool in self.tools],  # inside cwd only
            "permission_mode": "dontAsk",  # refuse every call not allowed above
            "setting_sources": [],  # no CLAUDE.md or user settings: the prompt alone
            "cwd": str(cwd),
            "output_format": {"type": "json_schema", "schema": schema},
        }


@dataclass(frozen=True)
class JudgeRun:
    """A judge's answer, what it cost, and the tool calls it made."""

    labels: list[dict]
    cost_usd: float | None = None
    usage: dict | None = None
    model_usage: dict | None = None
    claude_code_version: str | None = None
    tool_calls: list[dict] = field(default_factory=list)


async def judge(
    prompt: str, schema: dict, settings: JudgeSettings, cwd: Path, query: Query | None = None
) -> JudgeRun:
    """Run one judge on one prompt.

    Args:
        prompt: The rendered prompt.
        schema: JSON schema the answer must follow.
        settings: Model, effort and tools.
        cwd: Working directory of the agent.
        query: Sends the prompt; defaults to the Claude Agent SDK.

    Returns:
        The judge's labels, cost, token usage, harness version and tool calls.

    Raises:
        RuntimeError: If the agent ends without a valid structured answer.
    """
    result, harness, tool_calls = None, None, []
    async for message in (query or _sdk_query)(prompt, settings.options(schema, cwd)):
        if hasattr(message, "structured_output"):  # the final ResultMessage
            result = message
        elif getattr(message, "subtype", None) == "init":
            harness = message.data.get("claude_code_version")
        tool_calls += _tool_calls(message)
    if result is None or result.is_error or result.structured_output is None:
        raise RuntimeError(f"The judge returned no structured answer: {result!r}")
    return JudgeRun(
        labels=result.structured_output["labels"],
        cost_usd=result.total_cost_usd,
        usage=result.usage,
        model_usage=getattr(result, "model_usage", None),
        claude_code_version=harness,
        tool_calls=tool_calls,
    )


async def judge_many(
    prompts: list[str], schema: dict, settings: JudgeSettings, cwd: Path, query: Query | None = None
) -> list[JudgeRun | Exception]:
    """Run one judge per prompt, all at the same time; await it from a notebook.

    Args:
        prompts: The rendered prompts.
        schema: JSON schema the answers must follow.
        settings: Model, effort and tools.
        cwd: Working directory of the agents.
        query: Sends a prompt; defaults to the Claude Agent SDK.

    Returns:
        One run per prompt, in the same order, or the exception that stopped that judge.
    """
    runs = (judge(prompt, schema, settings, cwd, query) for prompt in prompts)
    return list(await asyncio.gather(*runs, return_exceptions=True))


def judge_all(
    prompts: list[str], schema: dict, settings: JudgeSettings, cwd: Path, query: Query | None = None
) -> list[JudgeRun | Exception]:
    """Run ``judge_many`` from synchronous code, such as the command line.

    Args:
        prompts: The rendered prompts.
        schema: JSON schema the answers must follow.
        settings: Model, effort and tools.
        cwd: Working directory of the agents.
        query: Sends a prompt; defaults to the Claude Agent SDK.

    Returns:
        One run per prompt, in the same order, or the exception that stopped that judge.
    """
    return asyncio.run(judge_many(prompts, schema, settings, cwd, query))


def check_labels(labels: list[dict], expected: Iterable[int]) -> None:
    """Check that a judge labelled each expected pair exactly once.

    Args:
        labels: The judge's answers.
        expected: Pair ids it was asked to label.

    Raises:
        ValueError: If a pair is missing, repeated or was not asked for.
    """
    given = [label["pair_id"] for label in labels]
    expected = sorted(expected)
    if sorted(given) != expected:
        missing = sorted(set(expected) - set(given))
        extra = sorted(set(given) - set(expected))
        raise ValueError(
            f"Labels do not match the batch: missing {missing}, extra {extra}, "
            f"{len(given)} labels for {len(expected)} pairs."
        )


def sdk_version() -> str | None:
    """Return the installed Claude Agent SDK version, or None when it is not installed."""
    try:
        return version("claude-agent-sdk")
    except PackageNotFoundError:
        return None


def _tool_calls(message: Any) -> list[dict]:
    """List the tool calls of an agent message, as ``{"name", "input"}``."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    return [
        {"name": block.name, "input": block.input}
        for block in content
        if hasattr(block, "name") and hasattr(block, "input")
    ]


def _sdk_query(prompt: str, options: dict[str, Any]) -> AsyncIterator[Any]:
    """Send a prompt through the Claude Agent SDK (dependency group ``audit``)."""
    # Imported here: the SDK is optional, and only judging needs it.
    from claude_agent_sdk import (  # pylint: disable=import-outside-toplevel,import-error
        ClaudeAgentOptions,
        query,
    )

    return query(prompt=prompt, options=ClaudeAgentOptions(**options))
