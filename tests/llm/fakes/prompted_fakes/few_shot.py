"""A fake fitted prompted package: few-shot prompts that show each problem's nearest examples.

It follows the prompted-method package contract of a fitted method: ``load``, ``llm_spec``,
``prompt_family`` and ``make_method``, and has no ``verify_extra``. Its method learns from dev
labels which problems may be examples, so its dev requests come from ``cross_requests``. Its
method folder keeps each example's sample id and the hash of the fields its prompts show, never
their text: ``load`` reads the texts from the dataset (``read_samples()``, so
``$CODEFORCES_DATA_DIR``) and checks the hashes (docs/adr/0004).
"""

import json
from pathlib import Path

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.data.loading import read_samples
from codeforces_tags.llm.prompted import PromptedLlm, load_prompted
from codeforces_tags.llm.prompts import LlmMethodSpec, field_sha256s, field_values, prompt_snapshot
from codeforces_tags.llm.settings import LlmSettings
from codeforces_tags.methods import MethodSpec
from codeforces_tags.preprocessing.pipeline import raw_to_problems

NAME = "fake-few-shot"
# How many examples a prompt shows, which of their fields, and the file of the method folder
# that pins them.
SHOTS = 2
EXAMPLE_FIELDS = ("prob_desc_description",)
EXAMPLES_FILE = "examples.json"


class NearestExamples(PromptedLlm):
    """A few-shot prompted method: each prompt shows the fit problems nearest to its problem.

    Its examples are the fit problems that have a focus tag; a problem's are the ``SHOTS`` of
    them whose sample ids are the nearest to its own.
    """

    def __init__(self, spec, prompt, examples=None):
        """Start with the examples given, as ``raw_to_problems`` gives them: none when unfitted."""
        super().__init__(spec, prompt)
        self.examples = examples

    def fit(self, raw_problems, labels):
        """Return the method fitted on some problems: its examples are those with a focus tag."""
        tagged = labels.index[labels[list(FOCUS_TAGS)].to_numpy().any(axis=1)]
        examples = raw_to_problems(raw_problems.loc[tagged])
        return NearestExamples(self.spec, self.prompt, examples.sort_index())

    def request_context(self, raw_problems):
        """Give each problem its nearest examples, with the fields its prompt shows."""
        shown = field_values(self.examples, EXAMPLE_FIELDS)
        return {
            int(sample_id): {
                "examples": [
                    {"sample_id": example, **shown[example]}
                    for example in self.nearest(int(sample_id))
                ]
            }
            for sample_id in raw_problems.index
        }

    def nearest(self, sample_id):
        """Return the sample ids of the examples nearest a problem, in sample id order."""
        examples = [int(example) for example in self.examples.index]
        by_distance = sorted(examples, key=lambda example: (abs(example - sample_id), example))
        return sorted(by_distance[:SHOTS])

    def save(self, folder):
        """Write ``prompted.json``, and each example's sample id and hash, into ``folder``."""
        super().save(folder)
        hashes = field_sha256s(self.examples, EXAMPLE_FIELDS)
        examples = [
            {"sample_id": sample_id, "sha256": sha256} for sample_id, sha256 in hashes.items()
        ]
        text = json.dumps({"examples": examples}, indent=2) + "\n"
        (Path(folder) / EXAMPLES_FILE).write_text(text, encoding="utf-8")


def load(folder):
    """Load the method saved in a method folder, reading its examples from the dataset.

    Raises:
        ValueError: If an example is not in the dataset, or its fields changed since it was saved.
    """
    prompted = load_prompted(folder)
    stored = json.loads((Path(folder) / EXAMPLES_FILE).read_text(encoding="utf-8"))
    hashes = {example["sample_id"]: example["sha256"] for example in stored["examples"]}
    problems = raw_to_problems(read_samples())
    examples = problems[problems.index.isin(list(hashes))].sort_index()
    if field_sha256s(examples, EXAMPLE_FIELDS) != hashes:
        raise ValueError(
            f"The examples of {folder} are not in the dataset, or their fields changed since "
            "the method was saved (docs/adr/0004)."
        )
    return NearestExamples(prompted.spec, prompted.prompt, examples)


def llm_spec(name):
    """Return the spec of the method: it shares its name with its prompt."""
    return LlmMethodSpec(
        name=name,
        prompt_name=NAME,
        template_path="few_shot.j2",
        schema_path=None,
        schema_name=None,
        fields=("prob_desc_description",),
        constants={},
        settings=LlmSettings(
            provider="openai",
            model="gpt-6-luna",
            reasoning_effort=None,
            temperature=None,
            top_p=None,
            max_output_tokens=256,
            output_format="text",
        ),
        all_tags=FOCUS_TAGS,
        extra={"shots": SHOTS},
    )


def prompt_family(prompt_name):
    """Return the one method of the prompt."""
    return (llm_spec(prompt_name),)


def make_method(name):
    """Return the method unfitted, with the prompt of this package.

    The fake is not in ``METHODS``: a registry of its own tells ``prompt_snapshot`` its package.
    """
    spec = llm_spec(name)
    registry = {name: MethodSpec(module=__name__, kind="fitted", requirements=())}
    return NearestExamples(spec, prompt_snapshot(spec, registry=registry))
