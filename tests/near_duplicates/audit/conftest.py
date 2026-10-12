"""Fixtures for the audit tests: a small dataset and a fake judge."""

import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from codeforces_tags.near_duplicates.audit import candidates, pipeline

# One pair per duplicate signal; every other problem has words of its own.
DESCRIPTIONS = {
    100: "This is the easy version. Given an array of n integers, find the maximum sum.",
    101: "This is the hard version. Given an array of n integers, find the maximum sum.",
    104: "Easy version: orchid quartz walnut ember falcon.",
    105: "Hard version: tundra violet saffron lagoon ivory.",
    106: "kappa lambda mu nu xi omicron pi rho sigma tau",
    107: "kappa lambda mu nu upsilon phi chi psi omega zeta",
    108: "abcdefghijkl mnopqrstuvwx",
    109: "abcdefghijklz mnopqrstuvwxz",
}
ROUNDS = {104: 7, 105: 7, 108: 3, 109: 4, 110: 5, 111: 6, 128: 1, 129: 2}  # round 0 otherwise
SAMPLE_INPUTS = {102: ["5"], 103: ["5"], 128: ["7"], 129: ["7"]}
CODE = {110: "for i in range(n): total += a[i]", 111: "for i in range(n): total += a[i]"}
# The pairs each signal flags, with the tier they fall into.
SIGNAL_PAIRS = {
    (100, 101): "A",  # easy and hard versions: word similarity 0.88
    (102, 103): "B",  # same round, identical sample input
    (104, 105): "B",  # same round, both announce a version
    (106, 107): "B",  # same round, word similarity 0.32
    (108, 109): "B",  # character similarity 0.84, different rounds
    (110, 111): "C",  # identical code, different rounds
    (128, 129): "C",  # identical sample input, different rounds
}


@pytest.fixture
def signal_pairs():
    """Return the pair each duplicate signal flags in ``problems``, with its tier."""
    return dict(SIGNAL_PAIRS)


@pytest.fixture
def problems():
    """Return 30 problems: one pair per duplicate signal, then unrelated problems of round 0."""
    ids = list(range(100, 130))
    return pd.DataFrame(
        {
            "prob_desc_description": [
                DESCRIPTIONS.get(k, f"alpha{k} beta{k} gamma{k} delta{k} epsilon{k}") for k in ids
            ],
            "prob_desc_input_spec": ["The first line contains n. " * 40] + [None] * (len(ids) - 1),
            "prob_desc_output_spec": ["Print the answer."] * len(ids),
            "prob_desc_created_at": [str(1600000000 + ROUNDS.get(k, 0)) for k in ids],
            "source_code": [CODE.get(k, f"print({k} * x_{k})") for k in ids],
            "prob_desc_sample_inputs": [SAMPLE_INPUTS.get(k, [f"x{k}"]) for k in ids],
            "prob_desc_sample_outputs": [["2"] if k in (100, 101) else [] for k in ids],
        },
        index=pd.Index(ids, name="sample_id"),
    )


@pytest.fixture
def small_audit(monkeypatch):
    """Shrink the audit's samples and batches to fit the small dataset."""
    monkeypatch.setattr(candidates, "AUDIT_CONTROL_SAMPLE", 5)
    monkeypatch.setattr(candidates, "AUDIT_WEAK_SIGNAL_SAMPLE", 2)
    monkeypatch.setattr(pipeline, "AUDIT_PAIRS_PER_JUDGE", 6)  # 2 batches of the 12 pairs


def labels_for(prompt):
    """Answer a prompt the way a judge would: label every pair it asks for."""
    listed = re.search(r"pair_ids: \[([\d,]*)\]", prompt)
    if listed:
        pair_ids = [int(x) for x in listed.group(1).split(",") if x]
    else:
        path = re.search(r"Read the JSON file (\S+) \(", prompt).group(1)
        with open(path, encoding="utf-8") as batch:
            pair_ids = [item["pair_id"] for item in json.load(batch)]
    return [
        {"pair_id": pair_id, "label": "versions" if pair_id == 0 else "different",
         "confidence": "low" if pair_id == 1 else "high", "reason": "test"}
        for pair_id in pair_ids
    ]  # fmt: skip


@pytest.fixture
def fake_query():
    """Return a query that answers like a judge, and records the options it received."""
    calls = []

    async def query(prompt, options):
        calls.append(options)
        yield SimpleNamespace(type="assistant")  # messages before the result are skipped
        yield SimpleNamespace(
            structured_output={"labels": labels_for(prompt)},
            is_error=False,
            total_cost_usd=0.01,
            usage={"output_tokens": 5},
        )

    query.calls = calls
    return query
