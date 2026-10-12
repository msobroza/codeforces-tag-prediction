"""What a prompted method asks: its spec, the snapshot of its prompt, and one request per problem.

A prompted method's package describes it with an ``LlmMethodSpec``: its prompt's template (and,
for the ``json`` output format, its schema), the problem fields and constants the template
reads, the settings that shape an answer, and the tags it chooses from. The template and schema
are either one versioned prompt file (``prompt_files``, ``templates/<prompt>.v<version>.yaml``)
or a ``.j2`` file and a ``.json`` file. ``prompt_snapshot`` reads them once, so a method folder
holds the prompt it was run with, and ``build_requests`` renders it for each problem, strictly
(``templating.render_string``).

A request's hash (``LlmRequest.sha256``) covers only what shapes the answer: the provider, the
model, the rendered prompt, the schema and the answer-shaping settings. It keys the method's
answers files (docs/adr/0003): a changed prompt never reuses an answer, and a transport
setting (``config.LLM_*``) never invalidates one. A schema is hashed as its JSON text, in its
own key order (``sha256_json`` sorts keys): structured outputs write an answer's keys in the
order of the schema's properties, so reordering them changes the answer.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from codeforces_tags.llm import providers
from codeforces_tags.llm.settings import LlmSettings
from codeforces_tags.methods import METHODS, MethodSpec
from codeforces_tags.prompt_files import read_prompt_file
from codeforces_tags.provenance import sha256_json
from codeforces_tags.templating import render_string


@dataclass(frozen=True)
class PromptSnapshot:
    """The prompt a prompted method was run with, as its method folder stores it.

    Attributes:
        name: The prompt's name in the prompt registry, e.g. ``"descode2tag"``.
        version: Its version in the registry, filled in by ``llm score``; None until then.
        uri: Its registry URI, e.g. ``"prompts:/descode2tag/2"``; None until then.
        content_sha256: ``prompt_content_sha256`` of the template, schema, settings and extra.
        template: The Jinja source of the prompt.
        response_format: The JSON schema of a ``json`` method's answer; None for ``text``.
    """

    name: str
    version: int | None
    uri: str | None
    content_sha256: str
    template: str
    response_format: dict | None

    def to_dict(self) -> dict[str, object]:
        """Return the snapshot as a JSON-ready dict: one key per field, in field order."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> PromptSnapshot:
        """Rebuild a snapshot from ``to_dict``'s output.

        Args:
            values: A value for every field, and no other key.

        Returns:
            The snapshot.

        Raises:
            TypeError: If a field is missing or a key is not a field.
        """
        return cls(**values)


@dataclass(frozen=True)
class LlmMethodSpec:  # pylint: disable=too-many-instance-attributes
    """What a prompted method's package says about one of its methods (``llm_spec(name)``).

    Attributes:
        name: The method's registry key, e.g. ``"descode2tag-json"``.
        prompt_name: Its prompt's name in the prompt registry, shared by the output formats of
            one setting, e.g. ``"descode2tag"``.
        template_path: The prompt's file, a POSIX path relative to the package of the method's
            registered module: a versioned prompt file (``.yaml``), or a ``.j2`` template.
        schema_path: The JSON schema's file, relative likewise, for the ``json`` output format
            (the prompt file itself, or a ``.json`` file beside a ``.j2`` template); None for
            ``text``.
        schema_name: The name the schema is sent under for ``json``; None for ``text``.
        fields: The problem columns the template reads, each under its column name.
        constants: Values the template reads under their keys, e.g. ``{"lang_cluster": ...}``.
        settings: The settings that shape an answer.
        all_tags: The tags the model chooses from, in order: the template reads them as
            ``all_tags``, and they are the method's output columns.
        extra: Anything else that defines the method's prompts (e.g. a few-shot method's number
            of examples); empty for a zero-shot method. JSON types only.
    """

    name: str
    prompt_name: str
    template_path: str
    schema_path: str | None
    schema_name: str | None
    fields: tuple[str, ...]
    constants: Mapping[str, str]
    settings: LlmSettings
    all_tags: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the spec; store its sequences as tuples and its mappings as copies.

        Raises:
            TypeError: If ``fields`` or ``all_tags`` is a single str.
            ValueError: If a ``json`` method lacks a schema path or name, if a ``text`` method
                has one, or if a path is absolute or leaves the method's package.
        """
        for name in ("fields", "all_tags"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise TypeError(
                    f"LlmMethodSpec.{name} must be a tuple of str, not the str {value!r}."
                )
            object.__setattr__(self, name, tuple(value))
        object.__setattr__(self, "constants", dict(self.constants))
        object.__setattr__(self, "extra", dict(self.extra))
        self._check_schema()
        for name in ("template_path", "schema_path"):
            _check_relative(name, getattr(self, name))

    def _check_schema(self) -> None:
        """Raise unless the method has a schema path and name exactly when it answers in JSON."""
        has_schema = (self.schema_path is not None, self.schema_name is not None)
        if self.settings.output_format == "json" and has_schema != (True, True):
            raise ValueError(
                f"{self.name} answers in JSON: it needs a schema_path and a schema_name."
            )
        if self.settings.output_format == "text" and has_schema != (False, False):
            raise ValueError(
                f"{self.name} answers in text: it sends no schema, so its schema_path and "
                "schema_name are None."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the spec as a JSON-ready dict: one key per field, the settings as a dict."""
        return asdict(self) | {"fields": list(self.fields), "all_tags": list(self.all_tags)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> LlmMethodSpec:
        """Rebuild a spec from ``to_dict``'s output, e.g. as read back from JSON.

        Args:
            values: A value for every field (``extra`` may be left out), and no other key.

        Returns:
            The spec.

        Raises:
            TypeError: If a field is missing or a key is not a field, here or in the settings.
            ValueError: As the constructor.
        """
        values = dict(values)
        settings = LlmSettings.from_dict(values.pop("settings"))
        return cls(settings=settings, **values)


@dataclass(frozen=True)
class LlmRequest:
    """One request of a prompted method: one problem's rendered prompt and the settings.

    Attributes:
        sample_id: The problem asked about.
        method: The prompted method's registry key.
        input: The rendered prompt, sent as one user message.
        settings: The settings that shape the answer.
        schema: The JSON schema of the answer, for the ``json`` output format; else None.
        schema_name: The name the schema is sent under; None without a schema.
    """

    sample_id: int
    method: str
    input: str
    settings: LlmSettings
    schema: dict | None
    schema_name: str | None

    def body(self) -> dict[str, object]:
        """Return the request as its provider's API takes it.

        Returns:
            ``request_body(self)`` of the adapter module of ``settings.provider``.

        Raises:
            ValueError: If the provider has no adapter (``llm.providers.PROVIDERS``).
        """
        return providers.provider_module(self.settings.provider).request_body(self)

    def sha256(self) -> str:
        """Hash what shapes the answer, and nothing else.

        The problem and the method are left out, as are the transport settings
        (``config.LLM_*``) and the output format, which the schema stands for.

        Returns:
            ``sha256_json`` of the provider, model, input, schema (its JSON text, in its own
            key order), schema name, temperature, top_p, reasoning effort and maximum output
            tokens.
        """
        return sha256_json(
            {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "input": self.input,
                "schema": _ordered_json(self.schema),
                "schema_name": self.schema_name,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "reasoning_effort": self.settings.reasoning_effort,
                "max_output_tokens": self.settings.max_output_tokens,
            }
        )


def prompt_content_sha256(
    template: str,
    response_format: Mapping[str, Any] | None,
    settings: LlmSettings,
    extra: Mapping[str, Any],
) -> str:
    """Hash what defines a prompted method's prompts: the one definition every check uses.

    Args:
        template: The Jinja source of the prompt.
        response_format: The JSON schema of a ``json`` method; None for ``text``.
        settings: The settings that shape an answer.
        extra: The spec's ``extra`` values.

    Returns:
        ``sha256_json`` of the four, under the keys ``template``, ``response_format`` (its JSON
        text, in its own key order), ``settings`` (a dict) and ``extra``.
    """
    return sha256_json(
        {
            "template": template,
            "response_format": _ordered_json(response_format),
            "settings": settings.to_dict(),
            "extra": extra,
        }
    )


def prompt_snapshot(
    spec: LlmMethodSpec, *, registry: Mapping[str, MethodSpec] = METHODS
) -> PromptSnapshot:
    """Read a prompted method's template and schema from its package.

    A ``.yaml`` template path is a versioned prompt file (``prompt_files.read_prompt_file``),
    which holds the prompt's name, its version, its output format, its template and, for
    ``json``, its schema (``response_format``); any other path is a Jinja template, beside a
    ``.json`` schema for ``json``.

    Args:
        spec: The method's spec; its paths are relative to the package of its registered
            module.
        registry: The registered methods.

    Returns:
        The snapshot, with ``version`` and ``uri`` None (``llm score`` fills them in).

    Raises:
        KeyError: If ``spec.name`` is not registered.
        FileNotFoundError: If the template or the schema is not in the package.
        ValueError: If a prompt file is not the spec's: not read as ``read_prompt_file`` reads
            it (a schema exactly when the spec answers in JSON), another prompt's, another
            output format's, or beside another schema file.
    """
    package = _package_of(registry[spec.name].module)
    if PurePosixPath(spec.template_path).suffix == ".yaml":
        template, response_format = _read_prompt_file(package, spec)
    else:
        template, response_format = _read_template_files(package, spec)
    return PromptSnapshot(
        name=spec.prompt_name,
        version=None,
        uri=None,
        content_sha256=prompt_content_sha256(template, response_format, spec.settings, spec.extra),
        template=template,
        response_format=response_format,
    )


def build_requests(
    problems: pd.DataFrame,
    spec: LlmMethodSpec,
    prompt: PromptSnapshot,
    context: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[LlmRequest]:
    """Render a prompted method's request for each problem.

    The template gets each column of ``spec.fields`` under its name, each constant under its
    key, ``all_tags``, and, with a ``context``, the problem's own variables (e.g. a few-shot
    method's ``examples``); nothing else. A missing field (NaN, None, empty or blank) is None,
    so it prints ``None``; any other value is printed as it is.

    Args:
        problems: The problems, as ``raw_to_problems`` returns them, indexed by ``sample_id``.
        spec: The method's spec.
        prompt: The method's prompt, from ``prompt_snapshot(spec)`` or its method folder.
        context: Extra variables of each problem, by ``sample_id``; every problem needs one.

    Returns:
        One request per problem, in the order of ``problems``.

    Raises:
        ValueError: If ``prompt`` is not the spec's (another name, or a content hash that its
            template, schema and the spec's settings and extra do not give), if two variables
            share a name, if a field is not a column of ``problems``, or if ``context`` lacks
            a problem.
        jinja2.UndefinedError: If the template reads a variable that was not given.
    """
    _check_prompt(spec, prompt)
    _check_variable_names(spec, context or {})
    rows = field_values(problems, spec.fields)
    contexts = _contexts(rows, context)
    shared = {**spec.constants, "all_tags": spec.all_tags}
    return [
        LlmRequest(
            sample_id=sample_id,
            method=spec.name,
            input=render_string(prompt.template, **(fields | shared | contexts[sample_id])),
            settings=spec.settings,
            schema=prompt.response_format,
            schema_name=spec.schema_name,
        )
        for sample_id, fields in rows.items()
    ]


def field_values(problems: pd.DataFrame, fields: Sequence[str]) -> dict[int, dict[str, object]]:
    """Return each problem's fields as a prompt shows them: a missing value as None.

    A value is missing when it is NaN, None, empty or blank. ``build_requests`` renders a method's
    fields from these values; a fitted prompted method renders its examples' fields from them too,
    and pins each example by their hash (``field_sha256s``).

    Args:
        problems: The problems, as ``raw_to_problems`` returns them, indexed by ``sample_id``.
        fields: The columns to read.

    Returns:
        Each problem's fields by name, by ``sample_id`` (a Python int), in the order of
        ``problems``.

    Raises:
        ValueError: If a field is not a column of ``problems``.
    """
    missing = [name for name in fields if name not in problems]
    if missing:
        raise ValueError(
            f"The problems lack the fields {missing}: pass raw_to_problems(raw_problems)."
        )
    rows = problems[list(fields)].to_dict(orient="index")
    return {
        int(sample_id): {name: _field_value(value) for name, value in row.items()}
        for sample_id, row in rows.items()
    }


def field_sha256s(problems: pd.DataFrame, fields: Sequence[str]) -> dict[int, str]:
    """Hash each problem's fields as a prompt shows them (``field_values``).

    A fitted prompted method that keeps problems as examples stores these hashes, never their
    text; its ``load`` reads the texts from the dataset and checks them (docs/adr/0004).

    Args:
        problems: The problems, as ``raw_to_problems`` returns them, indexed by ``sample_id``.
        fields: The columns to hash.

    Returns:
        ``sha256_json`` of each problem's fields, by ``sample_id``, in the order of ``problems``.

    Raises:
        ValueError: If a field is not a column of ``problems``.
    """
    return {
        sample_id: sha256_json(values)
        for sample_id, values in field_values(problems, fields).items()
    }


def _package_of(module: str) -> str:
    """Return the package a registered module's files are in: the module itself, or its parent.

    Python 3.11's ``importlib.resources.files`` takes a package only, where 3.12's also takes a
    plain module, e.g. ``codeforces_tags.methods.zero_shot.methods``.
    """
    imported = importlib.import_module(module)
    return module if hasattr(imported, "__path__") else imported.__spec__.parent


def _read_prompt_file(package: str, spec: LlmMethodSpec) -> tuple[str, dict | None]:
    """Return the template and schema of a spec's versioned prompt file, once checked."""
    if spec.schema_path not in (None, spec.template_path):
        raise ValueError(
            f"{spec.name} reads its prompt file {spec.template_path}: its schema_path must be "
            f"its prompt file, not {spec.schema_path}."
        )
    keys = {"output_format", "template"} | ({"response_format"} if spec.schema_path else set())
    prompt = read_prompt_file(package, spec.template_path, keys)
    if prompt.name != spec.prompt_name:
        raise ValueError(
            f"{spec.template_path} of {package} holds the prompt {prompt.name!r}, not "
            f"{spec.prompt_name!r}, the prompt of {spec.name}."
        )
    output_format = prompt.data["output_format"]
    if output_format != spec.settings.output_format:
        raise ValueError(
            f"{spec.template_path} of {package} answers in {output_format}, but {spec.name} "
            f"answers in {spec.settings.output_format}."
        )
    return prompt.data["template"], prompt.data.get("response_format")


def _read_template_files(package: str, spec: LlmMethodSpec) -> tuple[str, dict | None]:
    """Return a spec's Jinja template and, when it has one, its JSON schema file's schema."""
    folder = files(package)
    template = folder.joinpath(spec.template_path).read_text(encoding="utf-8")
    if spec.schema_path is None:
        return template, None
    return template, json.loads(folder.joinpath(spec.schema_path).read_text(encoding="utf-8"))


def _ordered_json(schema: Mapping[str, Any] | None) -> str | None:
    """Return a schema as compact JSON text in its own key order, or None without a schema.

    ``sha256_json`` sorts keys, which would lose the order of the schema's properties: the order
    in which a structured answer writes its keys, e.g. its reasoning before or after its tags.
    """
    if schema is None:
        return None
    return json.dumps(schema, separators=(",", ":"), allow_nan=False)


def _check_relative(name: str, path: str | None) -> None:
    """Raise unless a spec's path stays inside the method's package."""
    if path is None:
        return
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(
            f"LlmMethodSpec.{name} {path!r} must be a POSIX path relative to the method's "
            "package, inside it: the method folder stores it."
        )


def _check_prompt(spec: LlmMethodSpec, prompt: PromptSnapshot) -> None:
    """Raise unless the prompt is the spec's own."""
    content = prompt_content_sha256(
        prompt.template, prompt.response_format, spec.settings, spec.extra
    )
    if prompt.name != spec.prompt_name or prompt.content_sha256 != content:
        raise ValueError(
            f"The prompt {prompt.name!r} (content {prompt.content_sha256[:12]}) is not the prompt "
            f"of {spec.name}: build it with prompt_snapshot(spec)."
        )


def _check_variable_names(spec: LlmMethodSpec, context: Mapping[int, Mapping[str, Any]]) -> None:
    """Raise when two variables of the template would share a name."""
    context_keys = sorted({key for values in context.values() for key in values})
    sources = [
        *(("a field", name) for name in spec.fields),
        *(("a constant", name) for name in spec.constants),
        ("the tag list", "all_tags"),
        *(("a context key", name) for name in context_keys),
    ]
    seen: dict[str, str] = {}
    clashes = []
    for source, name in sources:
        if name in seen:
            clashes.append(f"{name!r} is both {seen[name]} and {source}")
        seen.setdefault(name, source)
    if clashes:
        raise ValueError(
            f"The variables of {spec.name}'s prompt need distinct names: {'; '.join(clashes)}."
        )


def _field_value(value: object) -> object:
    """Return a field as the prompt shows it: None when missing (NaN, None, empty or blank)."""
    if isinstance(value, str):
        return value if value.strip() else None
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else value


def _contexts(
    rows: Mapping[int, object], context: Mapping[int, Mapping[str, Any]] | None
) -> dict[int, dict[str, Any]]:
    """Return each problem's context variables by sample id: none without a context."""
    if context is None:
        return {sample_id: {} for sample_id in rows}
    missing = sorted(set(rows) - set(context))
    if missing:
        raise ValueError(
            f"The context lacks the problems {missing}: give every problem its variables."
        )
    return {sample_id: dict(context[sample_id]) for sample_id in rows}
