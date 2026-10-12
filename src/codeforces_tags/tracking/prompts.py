"""The prompt registry: each prompt's versions, registered once in a fixed order (docs/adr/0003).

``register_prompt_versions`` registers the versions of one prompt, given as plain data
(``PromptVersionContent``), once and in order; later, it only checks them. Its two callers are
adapters: ``register_prompts`` for the LLM toolkit's prompted methods, and the TypeSafe Jev
package's ``tracking`` module for the prompt its methods read. ``link_prompt_version`` links a
version to a logged model and to the run that logged it.

A prompt (e.g. ``descode2tag``) is shared by the prompted methods of one setting, one per output
format, and each method is one version of it. ``register_prompts`` registers a prompt's family,
the methods its package's ``prompt_family`` gives, in that order (``text`` then ``json`` for the
zero-shot settings), so a version number names the same prompt on every store. Each version
holds the template, the ``json`` method's schema as its ``response_format``, the settings as a
flat ``model_config`` dict, and the ``content_sha256`` tag. Once a prompt has versions, it only
checks them: a prompt version never changes after its first dev request, so a new wording is a
new version (a new prompt file, docs/adr/0009) or a new prompt, read by new methods.

Run records pin numeric versions (``prompts:/<name>/<version>``); the aliases that
``set_prompt_aliases`` sets, one per output format, mirror the frozen methods and gate nothing.
MLflow's ``PromptVersion.format`` is never called: prompts render through ``templating``
(docs/adr/0002, safeguard 4).

Versions are made with ``MlflowClient.create_prompt_version``, not ``register_prompt``: the
latter also copies each version's tags onto the prompt and links the prompt to the active
experiment from a thread of its own, which can outlive the command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities.model_registry import PromptVersion

from codeforces_tags.llm.prompts import LlmMethodSpec, PromptSnapshot, prompt_content_sha256
from codeforces_tags.tracking.environment import use_store

# The tag of each prompt version: the hash that names its content (for a prompted method, the
# prompt_content_sha256 of the method).
CONTENT_TAG = "content_sha256"
# The tag in which MLflow lists the prompt versions linked to a run or a logged model, as JSON.
LINKED_PROMPTS_TAG = "mlflow.linkedPrompts"
# How many hex digits of a hash an error message shows.
_SHORT_HASH = 12


@dataclass(frozen=True)
class PromptVersionContent:
    """What one version of a prompt holds, as plain data.

    Attributes:
        template: The template's text.
        response_format: The JSON schema of the answer; None for a free-text answer.
        model_config: The settings that shape an answer, as a flat dict.
        content_sha256: The hash that names the version's content: its ``content_sha256`` tag,
            which a later registration checks.
        description: The version's commit message.
        tags: The version's other tags; none is named ``content_sha256``.
    """

    template: str
    response_format: dict[str, Any] | None
    model_config: dict[str, Any]
    content_sha256: str
    description: str
    tags: Mapping[str, str] = field(default_factory=dict)


def register_prompt_versions(
    prompt_name: str, contents: Sequence[PromptVersionContent], **store: Any
) -> list[PromptVersion]:
    """Register a prompt's versions once, in the given order; later, only check them.

    When the registry has no version of the prompt, each content becomes a version, in order:
    version 1 is the first content, on every store. When it has some, they must be the first
    contents, in order (their ``content_sha256`` tags); the missing ones are then registered,
    which completes a registration that stopped early. It needs no API key: the registry is the
    MLflow store.

    Args:
        prompt_name: The prompt's name, e.g. ``"jev-tags"``.
        contents: Each version's content, in version order.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The prompt's versions, in version order.

    Raises:
        ValueError: If no content is given, or a content's tags name ``content_sha256``; or if
            the registry holds a version that is not one of the contents, in their order.
    """
    if not contents:
        raise ValueError(f"The prompt {prompt_name!r} has no version to register.")
    clashing = [number for number, content in enumerate(contents, 1) if CONTENT_TAG in content.tags]
    if clashing:
        raise ValueError(
            f"The tags of the versions {clashing} of {prompt_name!r} name {CONTENT_TAG}: "
            f"{CONTENT_TAG} is the version's content hash, given as its content_sha256."
        )
    use_store(**store)
    client = mlflow.MlflowClient()
    versions = _versions_of(client, prompt_name)
    _check_versions(prompt_name, versions, [content.content_sha256 for content in contents])
    for content in contents[len(versions) :]:
        versions.append(_register(client, prompt_name, content))
    return versions


def link_prompt_version(version: PromptVersion, model_id: str, **store: Any) -> None:
    """Link a prompt version to a logged model and to the run that logged it, once.

    MLflow lists the versions linked to a run or a model in its ``mlflow.linkedPrompts`` tag
    and adds a version only when the list lacks it, so linking again changes nothing.

    Args:
        version: The prompt version, e.g. as ``register_prompt_versions`` returns it.
        model_id: The logged model's id.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Raises:
        ValueError: If the model was logged outside a run; nothing is linked then.
    """
    use_store(**store)
    client = mlflow.MlflowClient()
    run_id = client.get_logged_model(model_id).source_run_id
    if run_id is None:
        raise ValueError(
            f"The model {model_id} was logged outside a run: its prompt {version.uri} is linked "
            "to a method's model and to the method run that logged it."
        )
    client.link_prompt_version_to_run(run_id, version)
    client.link_prompt_version_to_model(version.name, version.version, model_id)


def register_prompts(
    prompt_name: str, family: Sequence[tuple[LlmMethodSpec, PromptSnapshot]], **store: Any
) -> dict[str, PromptVersion]:
    """Register a prompt's versions once, in the family's order; later, only check them.

    When the registry has no version of the prompt, each method of the family becomes a
    version, in order: version 1 is the first method, on every store. When it has some, they
    must be the family's first versions, in order (their ``content_sha256`` tags); the missing
    ones are then registered, which completes a registration that stopped early. It needs no
    API key: the registry is the MLflow store.

    Args:
        prompt_name: The prompt's name, e.g. ``"descode2tag"``.
        family: Each method of the prompt with its snapshot (``prompt_snapshot(spec)``), in the
            order of its package's ``prompt_family(prompt_name)``.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        Each method's prompt version, by method name, in the family's order.

    Raises:
        ValueError: If the family is empty, names a method twice, or holds a method of another
            prompt or a snapshot that is not its spec's prompt; or if the registry holds a
            version that is not the family's, in its order.
    """
    hashes = _content_hashes(prompt_name, family)
    contents = [
        _method_content(spec, snapshot, sha256)
        for (spec, snapshot), sha256 in zip(family, hashes, strict=True)
    ]
    versions = register_prompt_versions(prompt_name, contents, **store)
    return {spec.name: version for (spec, _), version in zip(family, versions, strict=True)}


def set_prompt_aliases(
    prompt_name: str,
    versions: Mapping[str, PromptVersion],
    specs: Mapping[str, LlmMethodSpec],
    **store: Any,
) -> None:
    """Name each method's prompt version after its output format, e.g. ``text`` or ``json``.

    The aliases mirror the frozen methods; no command reads them, and run records pin numeric
    versions. Setting an alias again moves it to the version given.

    Args:
        prompt_name: The prompt's name.
        versions: Prompt versions by method name, as ``register_prompts`` returns them.
        specs: The spec of each method of ``versions``, by name.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Raises:
        ValueError: If a version has no spec, or is not a version of ``prompt_name``; no alias
            is set then.
    """
    without_spec = [name for name in versions if name not in specs]
    if without_spec:
        raise ValueError(f"The prompt versions of {without_spec} have no spec.")
    others = [name for name, version in versions.items() if version.name != prompt_name]
    if others:
        raise ValueError(f"The prompt versions of {others} are not versions of {prompt_name!r}.")
    use_store(**store)
    client = mlflow.MlflowClient()
    for name, version in versions.items():
        client.set_prompt_alias(prompt_name, specs[name].settings.output_format, version.version)


def _content_hashes(
    prompt_name: str, family: Sequence[tuple[LlmMethodSpec, PromptSnapshot]]
) -> list[str]:
    """Return the content hash of each method of a family, refusing a family that is not one."""
    if not family:
        raise ValueError(f"The family of the prompt {prompt_name!r} holds no method.")
    names = [spec.name for spec, _ in family]
    if len(set(names)) < len(names):
        raise ValueError(f"The family of the prompt {prompt_name!r} names a method twice: {names}.")
    return [_content_hash(prompt_name, spec, snapshot) for spec, snapshot in family]


def _content_hash(prompt_name: str, spec: LlmMethodSpec, snapshot: PromptSnapshot) -> str:
    """Return a method's content hash, refusing a method of another prompt or another snapshot."""
    if {spec.prompt_name, snapshot.name} != {prompt_name}:
        raise ValueError(
            f"{spec.name} belongs to the prompt {spec.prompt_name!r}, not {prompt_name!r} (its "
            f"snapshot is of {snapshot.name!r})."
        )
    content = prompt_content_sha256(
        snapshot.template, snapshot.response_format, spec.settings, spec.extra
    )
    if snapshot.content_sha256 != content:
        raise ValueError(
            f"The snapshot {snapshot.content_sha256[:_SHORT_HASH]} is not the prompt of "
            f"{spec.name}, whose template, schema, settings and extra hash to "
            f"{content[:_SHORT_HASH]}: build it with prompt_snapshot(spec)."
        )
    return content


def _versions_of(client: mlflow.MlflowClient, prompt_name: str) -> list[PromptVersion]:
    """Return every version of a prompt, by version number, creating the prompt when it is new."""
    if client.get_prompt(prompt_name) is None:
        client.create_prompt(prompt_name)
    return sorted(client.search_prompt_versions(prompt_name), key=lambda version: version.version)


def _check_versions(prompt_name: str, versions: Sequence[PromptVersion], hashes: list[str]) -> None:
    """Raise unless the registered versions are the given contents' first ones, in order."""
    numbers = [version.version for version in versions]
    found = [version.tags.get(CONTENT_TAG) for version in versions]
    if numbers != list(range(1, len(versions) + 1)) or found != hashes[: len(versions)]:
        raise ValueError(
            f"The prompt {prompt_name!r} holds the versions {numbers}, whose content hashes are "
            f"not the first ones of its versions, in order: {_short(found)} against "
            f"{_short(hashes)}. A prompt version never changes after its first dev request "
            "(docs/adr/0003, docs/adr/0009): a new wording is a new version or a new prompt, "
            "read by new methods under new names."
        )


def _method_content(
    spec: LlmMethodSpec, snapshot: PromptSnapshot, content_sha256: str
) -> PromptVersionContent:
    """Return a prompted method's prompt as the content of a version.

    The settings go in as a flat dict (MLflow leaves out a None value): ``PromptModelConfig``
    would drop the reasoning effort.
    """
    return PromptVersionContent(
        template=snapshot.template,
        response_format=snapshot.response_format,
        model_config=spec.settings.to_dict(),
        content_sha256=content_sha256,
        description=f"{spec.name}: the {spec.settings.output_format} output format",
    )


def _register(
    client: mlflow.MlflowClient, prompt_name: str, content: PromptVersionContent
) -> PromptVersion:
    """Register a content as the next version of a prompt."""
    return client.create_prompt_version(
        name=prompt_name,
        template=content.template,
        description=content.description,
        tags={CONTENT_TAG: content.content_sha256, **content.tags},
        response_format=content.response_format,
        model_config=content.model_config,
    )


def _short(hashes: Sequence[str | None]) -> list[str | None]:
    """Shorten hashes for an error message; a version without the tag shows None."""
    return [None if sha256 is None else sha256[:_SHORT_HASH] for sha256 in hashes]
