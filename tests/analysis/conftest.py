"""Fixtures for the analysis tests: a small dataset to measure splits on."""

import pandas as pd
import pytest


@pytest.fixture(name="problems")
def problems_fixture():
    """Return 10 problems over two years: 4 math, 1 graphs, 5 without a focus tag."""
    tags = [["math"]] * 4 + [["graphs"]] + [["greedy"]] * 5
    created = pd.to_datetime(["2020-01-01"] * 5 + ["2021-01-01"] * 5, utc=True)
    return pd.DataFrame(
        {"tags": tags, "prob_desc_created_at": created}, index=pd.Index(range(10), name="sample_id")
    )


@pytest.fixture(name="groups")
def groups_fixture():
    """Pair problems 0-1; every other problem is its own group."""
    return pd.Series([0, 0, 2, 3, 4, 5, 6, 7, 8, 9], index=pd.Index(range(10), name="sample_id"))


@pytest.fixture(name="many_problems")
def many_problems_fixture(problems):
    """Return 50 problems, five copies of ``problems``, enough for five stratified folds."""
    copies = pd.concat([problems] * 5, ignore_index=True)
    return copies.set_axis(pd.Index(range(50), name="sample_id"))


@pytest.fixture(name="many_groups")
def many_groups_fixture(many_problems):
    """Pair problems 0-1; every other problem is its own group."""
    group = pd.Series(many_problems.index, index=many_problems.index)
    group[1] = 0
    return group
