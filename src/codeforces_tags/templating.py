"""Render prompts with Jinja2, strictly: a variable the template needs and did not get is an error.

Every prompt of the project is rendered here, in one environment: the near-duplicate audit's
judge prompts, and the requests of every prompted method. The environment is strict
(``StrictUndefined``, so a missing variable raises instead of printing nothing) and escapes
nothing (a prompt is text, not HTML). A value is printed as it is and never rendered again, so
a solution that holds ``{{`` or ``{%`` reaches the prompt unchanged.

MLflow's ``PromptVersion.format`` is never used to render a prompt (docs/adr/0002, safeguard 4):
it renders with Jinja's lax ``Undefined`` (a missing variable becomes empty), has no loader, and
re-substitutes text inside values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jinja2


def render_template(name: str, template_dir: Path, /, **variables: object) -> str:
    """Render a template file of a folder.

    The template may ``{% include %}`` the other files of the folder, as the audit's prompts do.

    Args:
        name: The template's name without its extension: the file is ``<name>.j2``.
        template_dir: The folder holding the ``.j2`` templates.
        **variables: The values of the template, e.g. ``batch_file``. A variable may be called
            ``name`` or ``template_dir``, as ``name`` and ``template_dir`` are positional-only.

    Returns:
        The text, with the template's final newline dropped (Jinja's default).

    Raises:
        jinja2.TemplateNotFound: If the folder holds no ``<name>.j2``.
        jinja2.UndefinedError: If the template uses a variable that was not given.
    """
    environment = _environment(jinja2.FileSystemLoader(template_dir))
    return environment.get_template(f"{name}.j2").render(**variables)


def render_string(template: str, /, **variables: object) -> str:
    """Render a template given as a string.

    The environment has no loader, so the template cannot ``{% include %}`` or ``{% extends %}``
    another file: it renders on its own, as the snapshot of a prompt in a method folder must.

    Args:
        template: The template's source, e.g. the text of a ``.j2`` file.
        **variables: The values of the template. A variable may be called ``template``, as
            ``template`` is positional-only.

    Returns:
        The text, with the template's final newline dropped (Jinja's default).

    Raises:
        jinja2.TemplateSyntaxError: If the template is not valid Jinja.
        jinja2.UndefinedError: If the template uses a variable that was not given.
    """
    return _compiled(template).render(**variables)


def _environment(loader: jinja2.BaseLoader | None = None) -> jinja2.Environment:
    """Return the one kind of environment prompts are rendered in."""
    return jinja2.Environment(
        loader=loader,
        undefined=jinja2.StrictUndefined,
        autoescape=False,  # plain text, not HTML
    )


@lru_cache(maxsize=32)
def _compiled(template: str) -> jinja2.Template:
    """Compile a template once: it takes about a millisecond, and a run renders thousands."""
    return _environment().from_string(template)
