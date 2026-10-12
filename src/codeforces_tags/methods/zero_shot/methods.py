"""The package contract of the zero-shot methods: what the LLM toolkit's commands call.

``METHODS`` names this module for each of the four methods of ``settings.METHOD_NAMES``, and
``python -m codeforces_tags.llm`` serves them through it:

- ``load(folder)``: ``llm.prompted.load_prompted``, which reads a method folder's
  ``prompted.json`` back into a ``PromptedLlm``;
- ``llm_spec(name)``: a method's spec, from ``settings.py`` and this package's prompt files;
- ``prompt_family(prompt_name)``: the two methods of a setting, in the order their prompt
  versions are registered;
- ``verify_extra(raw_problems)``: the checks the two prompts rely on, which ``llm verify`` runs
  on every dev and test problem.

A method is its spec and nothing else: nothing is learned or chosen from data, and nothing
changes after its first dev request (docs/adr/0003).
"""

import pandas as pd

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.llm.answers import problem_list
from codeforces_tags.llm.prompted import load_prompted as load
from codeforces_tags.llm.prompts import LlmMethodSpec, build_requests, prompt_snapshot
from codeforces_tags.llm.settings import LlmSettings
from codeforces_tags.methods.zero_shot.settings import (
    FIELDS,
    LANGUAGE,
    MAX_OUTPUT_TOKENS,
    METHOD_NAMES,
    MODEL,
    OUTPUT_FORMATS,
    PROMPT_PATHS,
    PROVIDER,
    REASONING_EFFORT,
    SCHEMA_NAME,
    SETTINGS,
    TEMPERATURE,
    TOP_P,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems

__all__ = ["load", "llm_spec", "prompt_family", "verify_extra"]

# The fields that DesCode2Tag prints and Code2Tag does not, one per line, on lines 2 to 5 of its
# prompt: the description, the input and output specifications and the notes.
LINE_FIELDS = tuple(field for field in FIELDS["descode2tag"] if field not in FIELDS["code2tag"])


def llm_spec(name: str) -> LlmMethodSpec:
    """Return the spec of a zero-shot method.

    A method is a setting and an output format: ``descode2tag-json`` renders the DesCode2Tag
    prompt and asks for its answer as the JSON object of its schema, both read from the prompt
    file ``templates/descode2tag.v2.yaml``. The two methods of a setting render the same template
    from the same fields, so they send the same input.

    Args:
        name: One of ``METHOD_NAMES``, e.g. ``"code2tag-text"``.

    Returns:
        The spec: the setting's prompt file for the output format (``PROMPT_PATHS``), which is
        also the schema's file for ``json`` only, the schema's name for ``json`` only, the
        setting's fields, the language as the constant ``lang_cluster``, the request settings of
        ``settings.py``, the tags of ``config.ALL_TAGS``, and no ``extra``.

    Raises:
        ValueError: If ``name`` is not one of ``METHOD_NAMES``.
    """
    if name not in METHOD_NAMES:
        raise ValueError(f"{name!r} is not a zero-shot method: expected one of {METHOD_NAMES}.")
    setting, output_format = name.rsplit("-", 1)
    json_format = output_format == "json"
    prompt_path = PROMPT_PATHS[setting][output_format]
    return LlmMethodSpec(
        name=name,
        prompt_name=setting,
        template_path=prompt_path,
        schema_path=prompt_path if json_format else None,
        schema_name=SCHEMA_NAME if json_format else None,
        fields=FIELDS[setting],
        constants={"lang_cluster": LANGUAGE},
        settings=LlmSettings(
            provider=PROVIDER,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            output_format=output_format,
        ),
        all_tags=ALL_TAGS,
    )


def prompt_family(prompt_name: str) -> tuple[LlmMethodSpec, ...]:
    """Return the methods of a setting's prompt, in the order its versions are registered.

    ``register_prompts`` makes them versions 1 and 2 of the prompt on every store:
    ``prompts:/code2tag/1`` is ``code2tag-text`` and ``prompts:/code2tag/2`` is ``code2tag-json``.

    Args:
        prompt_name: A setting of ``SETTINGS``, which names its prompt.

    Returns:
        The specs of the setting's ``text`` and ``json`` methods, in ``OUTPUT_FORMATS`` order.

    Raises:
        ValueError: If ``prompt_name`` is not one of ``SETTINGS``.
    """
    if prompt_name not in SETTINGS:
        raise ValueError(
            f"{prompt_name!r} is not the prompt of a zero-shot setting: expected one of {SETTINGS}."
        )
    return tuple(llm_spec(f"{prompt_name}-{output_format}") for output_format in OUTPUT_FORMATS)


def verify_extra(raw_problems: pd.DataFrame) -> None:
    """Check, on raw problems, what the two approved prompts rely on.

    ``llm verify`` runs it on every dev and test problem. It preprocesses them itself
    (``raw_to_problems``), then checks that:

    1. no field of ``LINE_FIELDS`` holds a line break, i.e. a character at which
       ``str.splitlines`` splits (a line feed, a carriage return, a form feed, U+2028 and
       others), so that DesCode2Tag prints each field on its one labelled line;
    2. each problem's ``descode2tag-text`` input, with its lines 2 to 5 deleted, is its
       ``code2tag-text`` input. Lines end at line feeds alone: a solution with Windows line
       breaks keeps its carriage returns in both prompts.

    Args:
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``; other columns, such as
            the ``fold`` and ``group`` of both parts, pass through.

    Raises:
        ValueError: If a field of ``LINE_FIELDS`` holds a line break (naming the field and its
            problems), or if a DesCode2Tag input without its lines 2 to 5 is not the Code2Tag
            input (naming the problems).
    """
    problems = raw_to_problems(raw_problems)
    _check_one_line_fields(problems)
    _check_problem_lines(problems)


def _check_one_line_fields(problems: pd.DataFrame) -> None:
    """Raise unless every field of ``LINE_FIELDS`` fits on its one line of the prompt."""
    broken = {
        field: [sample_id for sample_id, text in problems[field].items() if _breaks_a_line(text)]
        for field in LINE_FIELDS
    }
    named = [f"{field} of the problems {problem_list(ids)}" for field, ids in broken.items() if ids]
    if named:
        raise ValueError(
            f"A line break in {'; '.join(named)} would move the next lines of the DesCode2Tag "
            "prompt: its problem fields would not be lines 2 to 5."
        )


def _check_problem_lines(problems: pd.DataFrame) -> None:
    """Raise unless deleting lines 2 to 5 of each DesCode2Tag input gives its Code2Tag input."""
    code2tag, descode2tag = (_text_inputs(problems, setting) for setting in SETTINGS)
    differ = [
        sample_id
        for sample_id, text in descode2tag.items()
        if _without_line_fields(text) != code2tag[sample_id]
    ]
    if differ:
        raise ValueError(
            f"The DesCode2Tag inputs of the problems {problem_list(differ)} are not their "
            "Code2Tag inputs once their lines 2 to 5 are deleted."
        )


def _text_inputs(problems: pd.DataFrame, setting: str) -> dict[int, str]:
    """Return each problem's input to the ``text`` method of a setting, by sample id."""
    spec = llm_spec(f"{setting}-text")
    requests = build_requests(problems, spec, prompt_snapshot(spec))
    return {request.sample_id: request.input for request in requests}


def _without_line_fields(text: str) -> str:
    """Delete the lines of ``LINE_FIELDS``, 2 to 5, from a prompt split at its line feeds."""
    lines = text.split("\n")
    return "\n".join(lines[:1] + lines[1 + len(LINE_FIELDS) :])


def _breaks_a_line(text: object) -> bool:
    """Tell whether a field holds a character at which ``str.splitlines`` splits."""
    return isinstance(text, str) and "".join(text.splitlines()) != text
