"""Fixtures of the zero-shot methods' tests: raw problems with every column, and a fake Ask.

No test of this folder sends a request or reads an API key. A test that predicts asks for the
fake Ask, which ``codeforces_tags.llm.client.ask_for`` then gives: a ``PromptedLlm`` without an
Ask of its own, as ``load`` returns it, calls ``ask_for`` each time it predicts.
"""

import json

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.llm import client

# The columns of the dataset that no prompt reads, beside those of make_raw_record.
OTHER_RAW_COLUMNS = {
    "prob_desc_input_from": "standard input",
    "prob_desc_output_to": "standard output",
    "lang_cluster": "Python",
    "exec_outcome": "PASSED",
    "hidden_unit_tests": "[]",
}
# A solution of more than 2,048 characters, the length at which promptsource cut every field:
# the prompt holds it whole.
LONG_SOLUTION = "s = input()\n" + "s = s.strip()  # keep every line\n" * 70 + "print(s[::-1])\n"


@pytest.fixture(name="zero_shot_raw")
def zero_shot_raw_fixture(make_raw_record):
    """Return three raw problems with every column of the dataset, as ``select_part`` gives them.

    They come out of sample id order, with their ``tags``, ``fold`` and ``group``. Problem 21's
    statement holds a glued sentence, an HTML entity, LaTeX and a notes heading, which
    ``raw_to_problems`` cleans or keeps; problem 7 has no input specification (NaN), a blank
    output specification, no notes (None, as ``read_samples`` gives a null) and a Python 2
    solution with Windows line breaks; problem 13's description holds Jinja syntax, and its
    solution is longer than 2,048 characters.
    """
    records = {
        21: make_raw_record(
            prob_desc_description="Count the leaves of the tree.Each leaf has degree $$$1$$$.",
            prob_desc_input_spec="The first line holds $$$n$$$ ($$$2 &lt; n \\le 10^5$$$).",
            prob_desc_output_spec="Print the number of leaves.",
            prob_desc_notes="NoteIn the first example, the tree has $$$2$$$ leaves.",
            source_code="# count the leaves\nn = int(input())\nprint(n - 1)\n",
            tags=["trees", "dfs and similar"],
        ),
        7: make_raw_record(
            prob_desc_description="Print the gcd of $$$a$$$ and $$$b$$$.",
            prob_desc_input_spec=np.nan,
            prob_desc_output_spec="   ",
            prob_desc_notes=None,
            source_code="import fractions\r\na, b = map(int, raw_input().split())\r\n"
            "print fractions.gcd(a, b)\r\n",
            lang="Python 2",
            difficulty=-1,
            tags=["math", "number theory"],
        ),
        13: make_raw_record(
            prob_desc_description="Reverse the string. Print {{ s }} and {% s %} as they are.",
            source_code=LONG_SOLUTION,
            difficulty=1600,
            tags=["strings"],
        ),
    }
    for sample_id, record in records.items():
        record |= OTHER_RAW_COLUMNS | {
            "src_uid": f"uid-{sample_id}",
            "code_uid": f"code-{sample_id}",
            "file_name": f"prog_{sample_id}.py",
        }
    raw = pd.DataFrame.from_dict(records, orient="index")
    raw.index.name = "sample_id"
    return raw.assign(fold=[0, 3, -1], group=[21, 7, 13])


@pytest.fixture(name="asked")
def asked_fixture():
    """Return the list of the requests that the fake Ask is sent, in order."""
    return []


@pytest.fixture(name="fake_ask")
def fake_ask_fixture(make_answer, zero_shot_raw, asked, monkeypatch):
    """Return a fake Ask, which ``client.ask_for`` gives for any settings while the test runs.

    It is a coroutine function: it reads the true tags of the request's problem from
    ``zero_shot_raw``, writes them as the request's output format asks (comma-separated, or the
    schema's JSON object), and appends the request to ``asked``. It sends nothing.
    """

    async def ask(request):
        asked.append(request)
        tags = zero_shot_raw.loc[request.sample_id, "tags"]
        answers = {"text": ", ".join(tags), "json": json.dumps({"tags": tags})}
        return make_answer(
            output_text=answers[request.settings.output_format],
            request_sha256=request.sha256(),
            sample_id=request.sample_id,
            method=request.method,
        )

    monkeypatch.setattr(client, "ask_for", lambda settings: ask)
    return ask
