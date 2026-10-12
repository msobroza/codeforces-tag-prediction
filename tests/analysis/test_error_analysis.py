"""Tests for the error analysis of a tag model's predictions."""

import pandas as pd
import pytest

from codeforces_tags.analysis.error_analysis import misclassified
from codeforces_tags.config import ERROR_EXCERPT_CHARS

# Four problems, out of index order. On graphs, 40 and 20 are false negatives, 10 and 30 false
# positives; math is right everywhere.
SAMPLE_IDS = pd.Index([40, 10, 30, 20], name="sample_id")


@pytest.fixture(name="statements")
def statements_fixture():
    """Return the four problems; problem 40's statement is longer than an excerpt."""
    return pd.DataFrame(
        {
            "tags": [["graphs"], ["math"], ["dp"], ["graphs", "trees"]],
            "statement": ["a" * (ERROR_EXCERPT_CHARS + 50), "b", "c", "d"],
        },
        index=SAMPLE_IDS,
    )


@pytest.fixture(name="truth")
def truth_fixture():
    """Return the true graphs and math labels of the four problems."""
    return pd.DataFrame({"graphs": [1, 0, 0, 1], "math": [0, 1, 0, 0]}, index=SAMPLE_IDS)


@pytest.fixture(name="predicted")
def predicted_fixture():
    """Return predictions that get graphs wrong on every problem."""
    return pd.DataFrame({"graphs": [0, 1, 1, 0], "math": [0, 1, 0, 0]}, index=SAMPLE_IDS)


@pytest.mark.parametrize(
    ("kind", "expected"), [("false positive", [10, 30]), ("false negative", [20, 40])]
)
def test_misclassified_lists_the_problems_with_that_error_in_index_order(
    statements, truth, predicted, kind, expected
):
    errors = misclassified(statements, truth, predicted, "graphs", kind)

    assert errors.index.tolist() == expected
    assert errors.columns.tolist() == ["tags", "statement"]


def test_misclassified_shows_the_tags_and_the_start_of_the_statement(statements, truth, predicted):
    errors = misclassified(statements, truth, predicted, "graphs", "false negative")

    assert errors.at[20, "tags"] == ["graphs", "trees"]
    assert errors.at[40, "statement"] == "a" * ERROR_EXCERPT_CHARS


def test_misclassified_returns_at_most_limit_problems(statements, truth, predicted):
    errors = misclassified(statements, truth, predicted, "graphs", "false positive", limit=1)

    assert errors.index.tolist() == [10]


def test_misclassified_finds_nothing_on_a_tag_predicted_right(statements, truth, predicted):
    errors = misclassified(statements, truth, predicted, "math", "false negative")

    assert errors.empty


def test_misclassified_rejects_an_unknown_kind_of_error(statements, truth, predicted):
    with pytest.raises(ValueError, match="'false positive', 'false negative'"):
        misclassified(statements, truth, predicted, "graphs", "true positive")
