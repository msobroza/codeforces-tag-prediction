"""A fake zero-shot prompted package: the prompt ``fake``, and one method per output format.

It follows the prompted-method package contract as ``methods/zero_shot/methods.py`` does:
``load``, ``llm_spec``, ``prompt_family`` and ``verify_extra``.
"""

import pandas as pd

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.llm.prompted import load_prompted as load
from codeforces_tags.llm.prompts import LlmMethodSpec
from codeforces_tags.llm.settings import OUTPUT_FORMATS, LlmSettings
from codeforces_tags.preprocessing.pipeline import raw_to_problems

__all__ = ["load", "llm_spec", "prompt_family", "verify_extra"]

PROMPT_NAME = "fake"


def llm_spec(name):
    """Return the spec of ``fake-text`` or ``fake-json``: the output format ends the name."""
    output_format = name.removeprefix(f"{PROMPT_NAME}-")
    json_format = output_format == "json"
    return LlmMethodSpec(
        name=name,
        prompt_name=PROMPT_NAME,
        template_path="tags.j2",
        schema_path="tags.schema.json" if json_format else None,
        schema_name="fake_tags" if json_format else None,
        fields=("prob_desc_description", "prob_desc_notes", "source_code"),
        constants={"language": "Python"},
        settings=LlmSettings(
            output_format=output_format,
            provider="openai",
            model="gpt-6-luna",
            max_output_tokens=512,
            reasoning_effort="none",
            temperature=0.3157894736842105,
            top_p=1.0,
        ),
        all_tags=ALL_TAGS,
    )


def prompt_family(prompt_name):
    """Return the methods of the prompt in registration order: ``text``, then ``json``."""
    return tuple(llm_spec(f"{prompt_name}-{output_format}") for output_format in OUTPUT_FORMATS)


def verify_extra(raw_problems: pd.DataFrame) -> None:
    """Fail unless every description fits on its one line of the prompt.

    Raises:
        ValueError: If a preprocessed description holds a line break.
    """
    descriptions = raw_to_problems(raw_problems)["prob_desc_description"]
    broken = sorted(
        int(sample_id) for sample_id in descriptions.index[descriptions.str.contains("\n")]
    )
    if broken:
        raise ValueError(f"The descriptions of the problems {broken} hold a line break.")
