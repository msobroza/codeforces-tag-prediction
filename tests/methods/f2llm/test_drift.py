"""Tests for measuring score_atol: the gap between the fit-time and the loaded paths' scores."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_info, threadpool_limits

from codeforces_tags.config import TUNING_BLAS_THREADS
from codeforces_tags.methods.f2llm import drift
from codeforces_tags.methods.f2llm.drift import score_gaps, setfit_score_gap, suggest_score_atol
from codeforces_tags.methods.f2llm.embedding import SOLUTION, embedding_columns, live_feature_frame
from codeforces_tags.methods.f2llm.encoder import EncoderConfig, PretrainedRecord, frozen_encoder
from codeforces_tags.methods.f2llm.estimators import make_probe
from codeforces_tags.methods.f2llm.method import F2LLMMethod
from codeforces_tags.methods.f2llm.setfit import SetFitEstimator
from codeforces_tags.methods.f2llm.settings import PROBE_GRID, SETFIT

# A probe configuration with both inputs, and SetFit's settings with a handful of pairs.
PROBE_PARAMS = next(p for p in PROBE_GRID if p["c"] == 1 and p["solution_weight"] == 0.5)
SMOKE_SETFIT = SETFIT | {"pairs_per_tag_positive": 2, "batch_pairs": 4, "micro_batch_pairs": 2}
# The held-out problems that both paths score.
HELD_OUT = 6
# The tolerance of the tiny model's CPU drift in these tests, whatever score_atol the registry
# holds: the atol command measures the GPU-to-CPU value written there, which may be smaller.
TINY_ATOL = 1e-4


class FirstColumnsScorer(BaseEstimator):
    """Scores each tag with one column of the frame, unchanged: the first ``n_tags`` columns."""

    def fit(self, X, y):
        """Remember the columns that score, and the rows it was fitted on."""
        self.columns_ = list(X.columns[: y.shape[1]])
        self.fit_index_ = X.index
        return self

    def decision_function(self, X):
        """Return the scoring columns."""
        return X[self.columns_].to_numpy()


def numbers(values, index):
    """Return a frame of two float columns, ``a`` and ``b``, with the given index."""
    return pd.DataFrame(np.asarray(values, dtype=float), index=index, columns=["a", "b"])


def blas_threads():
    """Return the thread counts of the BLAS libraries loaded in this process."""
    return {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}


def test_score_gaps_fits_once_and_returns_the_largest_score_difference():
    built = []

    def build():
        built.append(FirstColumnsScorer())
        return built[-1]

    fit_rows = numbers([[0, 0], [1, 1], [2, 2]], pd.Index([1, 2, 3], name="sample_id"))
    held_out = pd.Index([7, 8], name="sample_id")
    reference = numbers([[0.5, 0.25], [1.0, -1.0]], held_out)
    live = numbers([[0.5, 0.2501], [0.9997, -1.0]], held_out)

    gap = score_gaps(build, fit_rows, np.zeros((3, 2), dtype=np.int64), reference, live)

    assert gap == pytest.approx(3e-4)
    assert len(built) == 1
    assert built[0].fit_index_.equals(fit_rows.index)


def test_score_gaps_fits_with_blas_on_the_tuning_threads():
    observed = []

    class Recording(FirstColumnsScorer):
        """Records the BLAS threads it is fitted with."""

        def fit(self, X, y):
            """Record the threads, then fit."""
            observed.append(blas_threads())
            return super().fit(X, y)

    rows = numbers([[0, 0]], pd.Index([1], name="sample_id"))

    with threadpool_limits(limits=TUNING_BLAS_THREADS + 1, user_api="blas"):
        score_gaps(Recording, rows, np.zeros((1, 2), dtype=np.int64), rows, rows)
        after = blas_threads()

    assert observed == [{TUNING_BLAS_THREADS}]
    assert after == {TUNING_BLAS_THREADS + 1}  # the limit ends with the fit


def test_score_gaps_refuses_frames_of_other_problems():
    reference = numbers([[0, 0]], pd.Index([7], name="sample_id"))
    live = numbers([[0, 0]], pd.Index([8], name="sample_id"))

    with pytest.raises(ValueError, match="same problems in the same order"):
        score_gaps(FirstColumnsScorer, reference, np.zeros((1, 2)), reference, live)


def test_score_gaps_of_the_probe_between_a_batch_and_single_problems_is_within_score_atol(
    synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config
):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    fit_raw, held_out = synthetic_raw.iloc[HELD_OUT:], synthetic_raw.iloc[:HELD_OUT]
    fit_rows = live_feature_frame(fit_raw, encoder, encoder)
    batch = live_feature_frame(held_out, encoder, encoder)
    alone = pd.concat(
        [live_feature_frame(held_out.iloc[[row]], encoder, encoder) for row in range(HELD_OUT)]
    )
    labels = synthetic_labels.loc[fit_raw.index].to_numpy(dtype=np.int64)

    def build():
        return make_probe(PROBE_PARAMS, encoder.width)

    gap = score_gaps(build, fit_rows, labels, batch, alone)

    assert 0 <= gap < TINY_ATOL


def test_setfit_score_gap_compares_the_fitted_estimator_with_the_loaded_path(
    synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config
):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    fit_raw, held_out = synthetic_raw.iloc[HELD_OUT:], synthetic_raw.iloc[:HELD_OUT]
    fit_rows = live_feature_frame(fit_raw, encoder, encoder)
    reference = live_feature_frame(held_out, encoder, encoder)
    labels = synthetic_labels.loc[fit_raw.index].to_numpy(dtype=np.int64)
    shifted = reference.copy()
    solution = embedding_columns(SOLUTION, encoder.width)
    shifted[solution] = shifted[solution] + 0.5  # the head reads the frame's solution columns

    def build():
        return SetFitEstimator(
            head_params=PROBE_PARAMS,
            settings=SMOKE_SETFIT,
            device="cpu",
            pretrained=tiny_pretrained.to_dict(),
            encoder_config=tiny_config.to_dict(),
            resume=False,
        )

    same = setfit_score_gap(build, fit_rows, labels, reference, held_out)
    moved = setfit_score_gap(build, fit_rows, labels, shifted, held_out)

    assert 0 <= same < TINY_ATOL < moved


def test_setfit_score_gap_scores_the_loaded_method_one_problem_at_a_time(
    synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config, monkeypatch
):
    scored = []

    class Recording(F2LLMMethod):
        """Records the sample ids of each call to ``scores``."""

        def scores(self, raw_problems):
            """Record the problems, then score them."""
            scored.append(list(raw_problems.index))
            return super().scores(raw_problems)

    monkeypatch.setattr(drift, "F2LLMMethod", Recording)
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    fit_raw, held_out = synthetic_raw.iloc[HELD_OUT:], synthetic_raw.iloc[:HELD_OUT]
    fit_rows = live_feature_frame(fit_raw, encoder, encoder)
    reference = live_feature_frame(held_out, encoder, encoder)
    labels = synthetic_labels.loc[fit_raw.index].to_numpy(dtype=np.int64)

    def build():
        return SetFitEstimator(
            head_params=PROBE_PARAMS,
            settings=SMOKE_SETFIT,
            device="cpu",
            pretrained=tiny_pretrained.to_dict(),
            encoder_config=tiny_config.to_dict(),
            resume=False,
        )

    setfit_score_gap(build, fit_rows, labels, reference, held_out)

    assert scored == [[sample_id] for sample_id in held_out.index]


def test_setfit_score_gap_reads_missing_encoder_settings_as_the_estimator_does(
    synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config, monkeypatch
):
    # An estimator left without a pretrained record or an encoder configuration reads the
    # package's settings (the F2LLM-0.6B weights, which no test downloads); here they name the
    # tiny model instead.
    monkeypatch.setattr(EncoderConfig, "from_settings", classmethod(lambda cls: tiny_config))
    monkeypatch.setattr(PretrainedRecord, "from_settings", classmethod(lambda cls: tiny_pretrained))
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    fit_raw, held_out = synthetic_raw.iloc[HELD_OUT:], synthetic_raw.iloc[:HELD_OUT]
    fit_rows = live_feature_frame(fit_raw, encoder, encoder)
    reference = live_feature_frame(held_out, encoder, encoder)
    labels = synthetic_labels.loc[fit_raw.index].to_numpy(dtype=np.int64)

    def build():
        return SetFitEstimator(
            head_params=PROBE_PARAMS, settings=SMOKE_SETFIT, device="cpu", resume=False
        )

    gap = setfit_score_gap(build, fit_rows, labels, reference, held_out)

    assert 0 <= gap < TINY_ATOL


def test_setfit_score_gap_refuses_raw_problems_other_than_the_reference(synthetic_raw):
    reference = numbers([[0, 0]], pd.Index([-1], name="sample_id"))

    with pytest.raises(ValueError, match="same problems in the same order"):
        setfit_score_gap(FirstColumnsScorer, reference, np.zeros((1, 2)), reference, synthetic_raw)


@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        ([7.3e-6], 8e-5),  # rounded up, never down
        ([7e-8], 7e-7),  # 7e-7 / 1e-7 is 7.000000000000001 in floats: still 7e-7
        ([2e-7, 9.5e-6, 1e-6], 1e-4),  # the largest decides; 9.5e-5 rounds up to 1e-4
        ([0.0, 3e-7], 3e-6),
        ([1.2e-5], 2e-4),
    ],
)
def test_suggest_score_atol_is_ten_times_the_largest_gap_rounded_up_to_one_digit(gaps, expected):
    assert suggest_score_atol(gaps) == expected
    assert suggest_score_atol(iter(gaps)) == expected  # any iterable, read once


@pytest.mark.parametrize(
    ("gaps", "match"),
    [
        ([], "one finite, non-negative gap per method"),
        ([1e-6, -1e-6], "one finite, non-negative gap per method"),
        ([math.nan], "one finite, non-negative gap per method"),
        ([math.inf], "one finite, non-negative gap per method"),
        ([0.0, 0.0], "Every gap is 0"),
    ],
)
def test_suggest_score_atol_refuses_gaps_that_measure_nothing(gaps, match):
    with pytest.raises(ValueError, match=match):
        suggest_score_atol(gaps)
