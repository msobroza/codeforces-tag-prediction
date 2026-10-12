"""Tests for the handlers of a calibrated twin's commands, on synthetic dev problems.

The commands' gates and their runs end to end are tested in test_main.py, on a tiny dataset and
a fake Jev. These check what ``write_record`` builds the twin's record on: the base's dev scores
and the dev part's own folds and groups, which the record's tables are cross-fitted and
resampled on. git is faked: every file of the base is committed, at one commit.
"""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags import methods
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE, SCORES_FILE, stack_parts
from codeforces_tags.llm.store import answers_file
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev import platt, platt_commands
from codeforces_tags.methods.typesafe_jev.settings import PLATT_BASES, TUNING_FOLDER
from codeforces_tags.provenance import file_sha256
from codeforces_tags.splitting import PARTS

TWIN = "jev-statement-code-platt"
BASE = PLATT_BASES[TWIN]
# The one commit of every file of the base in these tests.
COMMIT = "c" * 40
# Two test problems, whose rows of the base's scores are NaN: no export reads them.
TEST_IDS = pd.Index([2000, 2001], name="sample_id")


@pytest.fixture(name="raw_dev")
def raw_dev_fixture(search_raw_dev):
    """Return the 100 dev problems of ``search_raw_dev``, in groups whose ids are no fold's.

    Problem ``1000 + p`` stays in fold ``p % 5`` and is in group ``10 + p % 25``: each group is
    inside one fold, and the ids of the groups (10 to 34) and of the folds (0 to 4) differ.
    """
    positions = np.arange(len(search_raw_dev))
    return search_raw_dev.assign(group=10 + positions % 25)


@pytest.fixture(name="problems")
def problems_fixture(raw_dev):
    """Return the tags of every problem: the dev problems', then the two test problems'."""
    test = pd.DataFrame({"tags": [["math"], ["graphs"]]}, index=TEST_IDS)
    return pd.concat([raw_dev[["tags"]], test])


@pytest.fixture(name="base_scores", autouse=True)
def base_scores_fixture(raw_dev, monkeypatch, tmp_path):
    """Write the base's frozen files in a folder of tmp_path, each committed; return its scores.

    Its dev probabilities are 0.3 on a negative and 0.7 on a positive, with noise, on Jev's 0.01
    grid, so that every Platt slope is above 0; its test rows are NaN. Its answers files and run
    record hold placeholders, which a twin's export names by sha256 and never reads.
    """
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(platt_commands, "committed_unchanged", lambda path: True)
    monkeypatch.setattr(platt_commands, "last_commit", lambda path: COMMIT)
    labels = focus_labels(raw_dev)
    noise = np.random.default_rng(7).normal(0.0, 0.2, labels.shape)
    dev = (0.3 + 0.4 * labels + noise).clip(0.01, 0.99).round(2).rename_axis(columns=None)
    scores = stack_parts(dev=dev, test=pd.DataFrame(np.nan, index=TEST_IDS, columns=dev.columns))
    directory = method_dir(BASE)
    directory.mkdir(parents=True)
    scores.to_csv(directory / SCORES_FILE)
    for path in (*(answers_file(BASE, part) for part in PARTS), directory / RUN_RECORD_FILE):
        path.write_text("{}\n", encoding="utf-8")
    return scores


def test_write_record_reads_the_splits_dev_folds_and_groups_and_the_bases_dev_scores(
    raw_dev, problems, base_scores, monkeypatch
):
    # Regression: a record built on the groups as its folds, and on the folds as its groups, passed
    # every test of the twins' commands, whose split puts each problem in a group of its own.
    handed = []
    monkeypatch.setattr(platt, "export_platt", lambda *arguments: handed.append(arguments))

    platt_commands.write_record(TWIN, raw_dev, problems)

    assert len(handed) == 1
    twin, out, dev, files = handed[0]
    assert (twin, out) == (TWIN, method_dir(TWIN) / TUNING_FOLDER)
    pd.testing.assert_series_equal(dev.folds, raw_dev["fold"])
    pd.testing.assert_series_equal(dev.groups, raw_dev["group"])
    pd.testing.assert_frame_equal(dev.labels, focus_labels(raw_dev))
    pd.testing.assert_frame_equal(dev.scores, base_scores.loc["dev"])  # no test row
    assert files == {
        name: {"sha256": file_sha256(path), "commit": COMMIT}
        for name, path in platt.base_paths(BASE).items()
    }


def test_export_writes_the_fold_thresholds_of_the_splits_five_dev_folds(
    raw_dev, problems, base_scores
):
    assert platt_commands.export(TWIN, raw_dev, problems) == 0

    tuning = method_dir(TWIN) / TUNING_FOLDER
    nested = pd.read_csv(tuning / platt.NESTED_FILE, index_col="fold")
    stability = pd.read_csv(tuning / platt.STABILITY_FILE, index_col=["tag", "rule"])
    folds = [f"fold_{fold}" for fold in range(5)]
    expected = platt.cross_fitted_rule_thresholds(
        "R2", base_scores.loc["dev"], focus_labels(raw_dev), raw_dev["fold"]
    )
    assert nested.index.tolist() == list(range(5))
    assert [column for column in stability.columns if column.startswith("fold_")] == folds
    found = stability.xs("R2", level="rule").loc[list(FOCUS_TAGS), folds]
    np.testing.assert_array_equal(found.to_numpy(), expected.T.to_numpy())
