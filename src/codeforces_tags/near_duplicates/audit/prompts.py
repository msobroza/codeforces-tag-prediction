"""Render the judges' prompts from their Jinja2 templates."""

from __future__ import annotations

import json
from pathlib import Path

from codeforces_tags.config import AUDIT_TEMPLATE_DIR
from codeforces_tags.templating import render_template

# One template per judging pass, in the order the passes run.
PASSES = ("first_pass", "second_pass_round_1", "second_pass_round_2")


def render_prompt(name: str, template_dir: Path = AUDIT_TEMPLATE_DIR, **variables: object) -> str:
    """Render a judge's prompt.

    Args:
        name: Template name without its extension, one of ``PASSES``.
        template_dir: Folder holding the ``.j2`` templates.
        **variables: Values for the template, e.g. ``batch_file``.

    Returns:
        The prompt text, exactly as a judge receives it.

    Raises:
        jinja2.UndefinedError: If the template uses a variable that was not given.
    """
    return render_template(name, template_dir, **variables)


def labels_schema(template_dir: Path = AUDIT_TEMPLATE_DIR) -> dict:
    """Read the JSON schema a judge's answer must follow.

    Args:
        template_dir: Folder holding ``labels.schema.json``.

    Returns:
        The schema: a ``labels`` list with ``pair_id``, ``label``, ``confidence``, ``reason``.
    """
    return json.loads((template_dir / "labels.schema.json").read_text(encoding="utf-8"))
