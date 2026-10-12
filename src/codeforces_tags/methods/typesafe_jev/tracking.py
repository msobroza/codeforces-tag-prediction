"""The TypeSafe Jev methods' prompt in MLflow's prompt registry (docs/adr/0003, docs/adr/0007).

The words Jev reads are the prompt ``jev-tags``: each of its versioned prompt files,
``templates/jev-tags.v<n>.yaml``, is its version n, registered once and in order on every store
by the generic ``tracking.prompts.register_prompt_versions``. A version holds the file's text as
its template (Jev answers probabilities, so it has no response format), the provider and the
model as its ``model_config``, the sha256 of the file as its ``content_sha256`` tag and
``sha256_json(SETTINGS)`` as its ``settings_sha256`` tag: the hash that each frozen method's
``jev.json`` holds. MLflow's ``PromptVersion.format`` is never called: the requests are built by
``questions``.

``run`` registers the prompt before it sends a request, as the LLM toolkit's ``run`` does, and
``freeze`` links the version its methods read to the method's logged model and method run. Its
methods are the two frozen ones: the searched method's words are its search's frozen record, not
a version of the prompt (docs/adr/0010), and ``search_tracking`` logs the search. This module
imports mlflow through ``codeforces_tags.tracking``, whose package sets MLflow's import-time
environment first; no prediction path imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeforces_tags.methods.typesafe_jev.settings import (
    FROZEN_NAMES,
    MODEL,
    PROMPT,
    PROMPT_NAME,
    PROMPT_VERSION,
    PROVIDER,
    SETTINGS,
)
from codeforces_tags.prompt_files import PromptFile
from codeforces_tags.provenance import sha256_json
from codeforces_tags.tracking.prompts import (
    PromptVersionContent,
    link_prompt_version,
    register_prompt_versions,
)

if TYPE_CHECKING:
    from mlflow.entities.model_registry import PromptVersion

# The tag of a version that holds sha256_json of the settings its methods' recipes hash.
SETTINGS_TAG = "settings_sha256"


def prompt_versions() -> list[PromptVersionContent]:
    """Return each version of the prompt ``jev-tags``, in version order: one per prompt file.

    Returns:
        The content of each version: version 1 is ``templates/jev-tags.v1.yaml``, which
        ``SETTINGS`` reads.
    """
    return [_content(PROMPT, SETTINGS)]


def register_jev_prompt(**store: Any) -> PromptVersion:
    """Register the versions of ``jev-tags`` once, in order; later, only check them.

    It needs no API key: the registry is the MLflow store.

    Args:
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The version that the methods read, ``settings.PROMPT_VERSION``.

    Raises:
        ValueError: If the registry holds a version of ``jev-tags`` that is not the version of
            its prompt file, in order (``register_prompt_versions``).
    """
    versions = register_prompt_versions(PROMPT_NAME, prompt_versions(), **store)
    return versions[PROMPT_VERSION - 1]


def link_jev_prompt(model_id: str, **store: Any) -> PromptVersion:
    """Register ``jev-tags``, then link the methods' version to a logged model and its run, once.

    Args:
        model_id: The method's logged model, as ``log_method`` returns it.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The version linked.

    Raises:
        ValueError: If the registry holds other versions of ``jev-tags``
            (``register_jev_prompt``), or if the model was logged outside a run
            (``link_prompt_version``).
    """
    version = register_jev_prompt(**store)
    link_prompt_version(version, model_id, **store)
    return version


def _content(prompt: PromptFile, settings: dict[str, object]) -> PromptVersionContent:
    """Return a prompt file as a version's content, tagged with the hash of its settings."""
    return PromptVersionContent(
        template=prompt.text,
        response_format=None,
        model_config={"provider": PROVIDER, "model": MODEL},
        content_sha256=prompt.sha256,
        description=(
            f"{prompt.name} v{prompt.version}: the words of {' and '.join(FROZEN_NAMES)}, "
            "one Noul question per focus tag"
        ),
        tags={SETTINGS_TAG: sha256_json(settings)},
    )
