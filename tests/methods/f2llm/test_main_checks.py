"""Tests for the checks that stop the F2LLM commands before they write or quote a wrong result.

On synthetic problems and the tiny encoder, as ``test_main.py``: nothing downloads.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.predictions import PREDICTIONS_FILE
from codeforces_tags.methods.f2llm import __main__ as cli
from codeforces_tags.methods.f2llm import setfit
from codeforces_tags.methods.f2llm.settings import (
    LATE_FUSION,
    LATE_FUSION_GRID,
    PROBE,
    PROBE_GRID,
    SETFIT,
    SETFIT_METHOD,
)
from codeforces_tags.methods.f2llm.tuning import OUT_OF_FOLD_FILE
from codeforces_tags.model.thresholds import CrossFittedTagThresholds

# The F2LLM lines' score_atol in the registry the commands read here, not the measured one.
SCORE_ATOL = 1e-4
# The probe configuration that probe_tuned exports in these tests, the first of the grid (a
# logistic regression at the smallest C, on the statement embedding alone), and another one, the
# last of the grid (a balanced ridge at the largest C, on both embeddings).
PROBE_PARAMS, OTHER_PROBE_PARAMS = PROBE_GRID[0], PROBE_GRID[-1]


@pytest.fixture(name="tiny_commands", autouse=True)
def tiny_commands_fixture(calls):
    """Point every command of these tests at the tiny encoder and at tmp_path, first (calls)."""
    return calls


def recorded_macro_f1(folder):
    """Return the out-of-fold macro F1 that a tuning/ folder records."""
    table = pd.read_csv(folder / OUT_OF_FOLD_FILE, index_col="metric", float_precision="round_trip")
    return table.at["macro_f1", "f1"]


def for_method(name, argv):
    """Return the probe's command arguments, for another method."""
    return [name if argument == PROBE else argument for argument in argv]


@pytest.mark.parametrize("shift", [1e-6, math.nan])
def test_freeze_refuses_a_refit_that_does_not_score_what_its_export_recorded(
    shift, probe_tuned, freeze_args, assets, calls
):
    path = probe_tuned / OUT_OF_FOLD_FILE
    table = pd.read_csv(path, index_col="metric", float_precision="round_trip")
    table.at["macro_f1", "f1"] += shift  # the code, labels or configuration changed since export
    table.to_csv(path)

    with pytest.raises(ValueError, match="that its export recorded in .* Nothing was written"):
        cli.main(freeze_args)

    assert not (assets / PROBE).exists()  # no method folder, no tuning/ copy, no prediction
    assert calls["log_method"] == []


@pytest.mark.usefixtures("frozen")  # the probe, frozen with PROBE_PARAMS
def test_freeze_of_late_fusion_reproduces_the_score_its_export_recorded_with_the_frozen_probe(
    write_tuning, freeze_args, assets, capsys
):
    folder = write_tuning(LATE_FUSION, LATE_FUSION_GRID[0])  # tuned with the frozen probe
    capsys.readouterr()

    assert cli.main(for_method(LATE_FUSION, freeze_args)) == 0

    assert capsys.readouterr().out.splitlines()[0] == (
        f"{LATE_FUSION}: the refit scores dev out of fold at macro F1 "
        f"{recorded_macro_f1(folder):.4f}, as {folder / OUT_OF_FOLD_FILE} records"
    )
    assert (assets / LATE_FUSION / PREDICTIONS_FILE).exists()


@pytest.mark.usefixtures("frozen")  # the probe, frozen with PROBE_PARAMS
def test_freeze_of_late_fusion_refuses_a_weight_tuned_with_another_probe_than_the_frozen_one(
    write_tuning, freeze_args, assets
):
    write_tuning(PROBE, OTHER_PROBE_PARAMS)  # the probe, exported again after its freeze
    write_tuning(LATE_FUSION, LATE_FUSION_GRID[0])  # late fusion, tuned and exported with it

    with pytest.raises(ValueError, match="that its export recorded in .* Nothing was written"):
        cli.main(for_method(LATE_FUSION, freeze_args))

    assert not (assets / LATE_FUSION).exists()


@pytest.mark.usefixtures("frozen")  # the probe, whose configuration SetFit's head takes
def test_freeze_of_setfit_has_no_export_to_reproduce_and_no_tuning_folder_to_copy(
    freeze_args, assets, tmp_path, monkeypatch, capsys
):
    tiny = SETFIT | {"pairs_per_tag_positive": 1, "batch_pairs": 4, "micro_batch_pairs": 2}
    monkeypatch.setattr(cli, "SETFIT", tiny)  # a few steps on the tiny encoder, on CPU
    monkeypatch.setattr(setfit, "experiments_dir", lambda name: tmp_path / "experiments" / name)
    capsys.readouterr()

    assert cli.main(for_method(SETFIT_METHOD, freeze_args)) == 0

    printed = capsys.readouterr().out.splitlines()
    assert not [line for line in printed if "the refit scores dev out of fold" in line]
    assert f"{SETFIT_METHOD} tunes nothing on dev: it has no tuning/ folder" in printed
    assert (assets / SETFIT_METHOD / PREDICTIONS_FILE).exists()


def not_a_number(problems):
    """Stand in for a loaded method's scores: NaN for every problem and tag."""
    return pd.DataFrame(np.nan, index=problems.index, columns=list(FOCUS_TAGS))


@pytest.mark.usefixtures("probe_tuned")
def test_freeze_refuses_a_method_folder_whose_live_scores_are_not_numbers(
    freeze_args, assets, monkeypatch
):
    monkeypatch.setattr(
        cli, "load_method", lambda _folder, **_registry: SimpleNamespace(scores=not_a_number)
    )

    with pytest.raises(ValueError, match="dev problems up to nan away from the refit"):
        cli.main(freeze_args)

    assert not (assets / PROBE / PREDICTIONS_FILE).exists()


class NotANumberFit(CrossFittedTagThresholds):
    """Scores the first tag of every dev problem NaN out of fold, once fitted."""

    def fit(self, X, y, *, groups):
        """Fit, then make the first tag's out-of-fold scores NaN."""
        super().fit(X, y, groups=groups)
        self.oof_scores_ = self.oof_scores_.copy()
        self.oof_scores_[:, 0] = np.nan
        return self


@pytest.mark.usefixtures("frozen")
def test_verify_retrain_refuses_a_refit_whose_scores_are_not_numbers(verify_args, monkeypatch):
    monkeypatch.setattr(cli, "CrossFittedTagThresholds", NotANumberFit)

    with pytest.raises(ValueError, match="dev, refit from the cache: .* by up to nan"):
        cli.main([*verify_args, "--retrain"])
