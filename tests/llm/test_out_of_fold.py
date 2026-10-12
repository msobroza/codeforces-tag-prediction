"""Tests for the out-of-fold requests of prompted methods that learn from dev labels."""

import dataclasses

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.out_of_fold import cross_requests
from codeforces_tags.llm.prompted import PromptedLlm
from codeforces_tags.methods.label_prior.method import LabelPrior

# A few-shot prompt: the sample ids of its examples, then the problem.
TEMPLATE = 'Examples: {{ examples | join(", ") }}\nTag: {{ prob_desc_description }}'
SAMPLE_IDS = list(range(201, 210))
# The problems labelled math: 201 and 207 in fold 0, 202 and 205 in fold 1, 209 in fold 2.
MATH = [1, 1, 0, 0, 1, 0, 1, 0, 1]


class FewShotFake(PromptedLlm):
    """A fitted prompted method: its prompts show, as examples, the fit problems tagged math."""

    def __init__(self, spec, prompt, examples=()):
        """Start with the examples given: none for an unfitted method."""
        super().__init__(spec, prompt)
        self.examples = tuple(examples)

    def fit(self, raw_problems, labels):
        """Return the method fitted on some problems: its examples are those labelled math."""
        assert labels.index.equals(raw_problems.index)
        examples = [int(sample_id) for sample_id in labels.index[labels["math"] == 1]]
        return FewShotFake(self.spec, self.prompt, examples)

    def request_context(self, raw_problems):
        """Give every problem the method's examples."""
        return {int(sample_id): {"examples": self.examples} for sample_id in raw_problems.index}


@pytest.fixture(name="make_method")
def make_method_fixture(make_llm_spec, snapshot_of):
    """Return a builder of the unfitted few-shot method; the methods it built are listed."""
    spec = make_llm_spec(name="few-shot-fake", fields=("prob_desc_description",))
    prompt = snapshot_of(spec, TEMPLATE)
    built = []

    def make():
        built.append(FewShotFake(spec, prompt))
        return built[-1]

    make.built = built
    return make


@pytest.fixture(name="nine_problems")
def nine_problems_fixture(make_raw_record):
    """Return nine raw problems, indexed by sample id."""
    records = [make_raw_record(prob_desc_description=f"Problem {i}.") for i in SAMPLE_IDS]
    return pd.DataFrame(records, index=pd.Index(SAMPLE_IDS, name="sample_id"))


@pytest.fixture(name="labels")
def labels_fixture(nine_problems):
    """Return 0/1 focus-tag labels: math on five problems, no other tag."""
    labels = pd.DataFrame(0, index=nine_problems.index, columns=list(FOCUS_TAGS))
    labels["math"] = MATH
    return labels


@pytest.fixture(name="folds")
def folds_fixture(nine_problems):
    """Assign the problems to folds 0, 1, 2, 0, 1, 2, ..."""
    return pd.Series([0, 1, 2] * 3, index=nine_problems.index, name="fold")


def test_cross_requests_shows_each_fold_the_examples_of_the_other_folds(
    make_method, nine_problems, labels, folds
):
    requests = cross_requests(make_method, nine_problems, labels, folds)

    examples = {request.sample_id: request.input.splitlines()[0] for request in requests}
    assert examples == {
        201: "Examples: 202, 205, 209",
        202: "Examples: 201, 207, 209",
        203: "Examples: 201, 202, 205, 207",
        204: "Examples: 202, 205, 209",
        205: "Examples: 201, 207, 209",
        206: "Examples: 201, 202, 205, 207",
        207: "Examples: 202, 205, 209",
        208: "Examples: 201, 207, 209",
        209: "Examples: 201, 202, 205, 207",
    }


def test_cross_requests_are_the_requests_of_a_method_fitted_on_the_other_folds(
    make_method, nine_problems, labels, folds
):
    requests = cross_requests(make_method, nine_problems, labels, folds)

    assert len(make_method.built) == 3
    by_problem = {request.sample_id: request for request in requests}
    for fold, method in enumerate(make_method.built):
        held_out = folds == fold
        fitted = method.fit(nine_problems[~held_out], labels[~held_out])
        for request in fitted.requests(nine_problems[held_out]):
            assert by_problem[request.sample_id] == request


def test_cross_requests_follow_the_order_of_the_problems(make_method, nine_problems, labels, folds):
    order = [205, 201, 209, 203, 207, 202, 208, 204, 206]

    requests = cross_requests(
        make_method, nine_problems.loc[order], labels.loc[order], folds.loc[order]
    )

    assert [request.sample_id for request in requests] == order


def test_cross_requests_checks_the_folds_as_cross_predict_does(
    make_method, nine_problems, labels, folds
):
    with pytest.raises(ValueError, match="negative fold"):
        cross_requests(make_method, nine_problems, labels, folds.where(folds != 2, -1))
    with pytest.raises(ValueError, match="labels must have the same index"):
        cross_requests(make_method, nine_problems, labels.iloc[::-1], folds)
    assert make_method.built == []


def test_cross_requests_refuses_a_method_with_nothing_to_fit(
    make_llm_spec, snapshot_of, nine_problems, labels, folds
):
    spec = make_llm_spec()
    zero_shot = PromptedLlm(spec, snapshot_of(spec, TEMPLATE))

    with pytest.raises(TypeError, match=r"PromptedLlm has no fit\(raw_problems, labels\)"):
        cross_requests(lambda: zero_shot, nine_problems, labels, folds)


def test_cross_requests_refuses_a_fitted_method_that_builds_no_requests(
    nine_problems, labels, folds
):
    with pytest.raises(TypeError, match=r"LabelPrior has no requests\(raw_problems\)"):
        cross_requests(LabelPrior, nine_problems, labels, folds)


def test_cross_requests_refuses_folds_fitted_into_different_methods(
    make_method, nine_problems, labels, folds
):
    names = iter(["few-shot-a", "few-shot-b", "few-shot-c"])

    class Renamed(FewShotFake):
        """A few-shot method whose every fit gives a method of another name."""

        def fit(self, raw_problems, labels):
            """Return the method fitted on some problems, under the next name."""
            fitted = super().fit(raw_problems, labels)
            return FewShotFake(dataclasses.replace(fitted.spec, name=next(names)), fitted.prompt)

    def make_renamed():
        method = make_method()
        return Renamed(method.spec, method.prompt)

    with pytest.raises(ValueError, match="one method"):
        cross_requests(make_renamed, nine_problems, labels, folds)
