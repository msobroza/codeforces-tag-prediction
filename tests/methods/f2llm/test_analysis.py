"""Tests for the shared analysis files of the f2llm methods."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.f2llm.analysis import (
    INPUT_LENGTHS_FILE,
    PAIR_COUNTS_FILE,
    PAIRS_FILE,
    TFIDF_REFERENCE_FILE,
    embedding_pairs,
    input_lengths,
    tfidf_reference,
    write_or_check,
)
from codeforces_tags.methods.f2llm.embedding import SOLUTION, STATEMENT, encoder_inputs
from codeforces_tags.methods.f2llm.encoder import load_tokenizer, token_counts
from codeforces_tags.methods.f2llm.settings import PAIRS_QUANTILE
from codeforces_tags.model.training import build_pipeline, cross_fit, fit_tag_model
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The real dataset, for the size cap of the pairs file: its problems and known pairs.
DATASET_PROBLEMS, DATASET_KNOWN_PAIRS = 4982, 82


def make_split():
    """Return a split of 12 problems: dev folds 0-2 (3 each) and 3 test problems.

    Sample ids are 20-31, out of order. Problems 20 and 21 (dev fold 0) and 30 and 31 (test)
    are two near-duplicate groups; every other problem has its own group.
    """
    sample_ids = [25, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31]
    folds = {20: 0, 21: 0, 22: 0, 23: 1, 24: 1, 25: 1, 26: 2, 27: 2, 28: 2}
    return pd.DataFrame(
        {
            "part": ["dev" if sample_id in folds else "test" for sample_id in sample_ids],
            "fold": [folds.get(sample_id, -1) for sample_id in sample_ids],
            "group": [{21: 20, 31: 30}.get(sample_id, sample_id) for sample_id in sample_ids],
        },
        index=pd.Index(sample_ids, name="sample_id"),
    )


def make_vectors(split):
    """Return random statement vectors of the split's problems, with three planted twins.

    22 (fold 0) and 23 (fold 1) are a cross-fold twin, 24 (dev) and 29 (test) a dev-test twin,
    and the known pair 20-21 is a twin too. The rows come in another order than the split's.
    """
    rng = np.random.default_rng(0)
    vectors = pd.DataFrame(rng.normal(size=(len(split), 8)), index=split.index)
    for original, twin in ((22, 23), (24, 29), (20, 21)):
        vectors.loc[twin] = vectors.loc[original] + rng.normal(scale=0.01, size=8)
    return vectors.iloc[::-1]


def relation(first, second):
    """Return how two problems of the split sit, as embedding_pairs names it ("none": neither)."""
    if first["group"] == second["group"]:
        return "same_group"
    if first["part"] != second["part"]:
        return "dev_test"
    if first["part"] == "dev" and first["fold"] != second["fold"]:
        return "cross_fold"
    return "none"


def brute_force_pairs(vectors, split):
    """Return every pair (a < b) with its cosine and relation, one pair at a time."""
    unit = vectors.div(np.linalg.norm(vectors, axis=1), axis=0)
    rows = [
        (a, b, float(unit.loc[a] @ unit.loc[b]), relation(split.loc[a], split.loc[b]))
        for a, b in itertools.combinations(sorted(split.index), 2)
    ]
    return pd.DataFrame(rows, columns=["sample_id_a", "sample_id_b", "cosine", "relation"])


def test_write_or_check_writes_a_table_once_then_checks_it(tmp_path):
    frame = pd.DataFrame({"sample_id": [3, 1], "cosine": [0.5, 0.25], "kind": ["known", "flagged"]})
    path = tmp_path / "f2llm" / PAIRS_FILE

    saved = write_or_check(frame, path)
    text = path.read_text(encoding="utf-8")
    checked = write_or_check(frame.assign(cosine=frame["cosine"] + 1e-10), path)

    assert saved == f"saved 2 rows to {path}"
    assert checked == f"{path} holds the same 2 rows"
    assert text.splitlines() == ["sample_id,cosine,kind", "3,0.5,known", "1,0.25,flagged"]
    assert [file.name for file in path.parent.iterdir()] == [PAIRS_FILE]


@pytest.mark.parametrize(
    "change",
    [
        lambda frame: frame.assign(cosine=frame["cosine"] + 1e-6),  # a float beyond atol
        lambda frame: frame.assign(kind=["known", "known"]),  # another string
        lambda frame: frame.assign(sample_id=[3, 2]),  # another integer
        lambda frame: frame[["cosine", "sample_id", "kind"]],  # another column order
        lambda frame: frame.iloc[:1],  # a row less
        lambda frame: frame.assign(cosine=[0.5, np.nan]),  # NaN for a number
    ],
)
def test_write_or_check_refuses_another_table_and_leaves_the_file(tmp_path, change):
    frame = pd.DataFrame({"sample_id": [3, 1], "cosine": [0.5, 0.25], "kind": ["known", "flagged"]})
    path = tmp_path / PAIRS_FILE
    write_or_check(frame, path)
    text = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the recomputed table"):
        write_or_check(change(frame), path)
    assert path.read_text(encoding="utf-8") == text


def test_write_or_check_takes_nan_as_nan_and_its_own_tolerance(tmp_path):
    frame = pd.DataFrame({"cosine": [0.5, np.nan]})
    path = tmp_path / PAIRS_FILE
    write_or_check(frame, path)

    assert write_or_check(frame, path) == f"{path} holds the same 2 rows"
    assert write_or_check(frame.assign(cosine=[0.6, np.nan]), path, atol=0.2).endswith("rows")


def test_input_lengths_counts_the_untruncated_tokens_of_each_encoder_input(
    synthetic_raw, tiny_pretrained, tiny_config, tmp_path
):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)
    raw = synthetic_raw.iloc[:5]

    lengths = input_lengths(raw, tokenizer, tiny_config)

    inputs = encoder_inputs(raw_to_problems(raw), tiny_config)
    assert list(lengths.columns) == ["sample_id", "statement_tokens", "solution_tokens"]
    assert (lengths.dtypes == "int64").all()
    assert lengths["sample_id"].tolist() == raw.index.tolist()
    np.testing.assert_array_equal(
        lengths["statement_tokens"], token_counts(inputs[STATEMENT], tokenizer)
    )
    np.testing.assert_array_equal(
        lengths["solution_tokens"], token_counts(inputs[SOLUTION], tokenizer)
    )
    assert lengths["statement_tokens"].min() > tiny_config.max_length  # not truncated
    path = tmp_path / INPUT_LENGTHS_FILE
    assert write_or_check(lengths, path).startswith("saved 5 rows")
    assert write_or_check(input_lengths(raw, tokenizer, tiny_config), path).endswith("rows")


def test_embedding_pairs_lists_the_known_pairs_and_flags_the_top_candidates():
    split = make_split()
    vectors = make_vectors(split)

    pairs, counts = embedding_pairs(vectors, split, quantile=0.95)

    every = brute_force_pairs(vectors, split)
    candidates = every[every["relation"].isin(["cross_fold", "dev_test"])]
    cut = np.quantile(candidates["cosine"], 0.95)
    flagged = candidates[candidates["cosine"] >= cut]
    known = every[every["relation"] == "same_group"]
    assert list(pairs.columns) == ["sample_id_a", "sample_id_b", "cosine", "kind", "relation"]
    assert pairs["kind"].tolist() == ["known"] * 2 + ["flagged"] * len(flagged)
    assert pairs[["sample_id_a", "sample_id_b"]].values.tolist() == [
        *known[["sample_id_a", "sample_id_b"]].values.tolist(),
        *flagged[["sample_id_a", "sample_id_b"]].values.tolist(),
    ]
    assert pairs["relation"].tolist() == [*known["relation"], *flagged["relation"]]
    np.testing.assert_allclose(pairs["cosine"], [*known["cosine"], *flagged["cosine"]], atol=1e-12)
    assert {(22, 23, "cross_fold"), (24, 29, "dev_test")} <= set(
        zip(pairs["sample_id_a"], pairs["sample_id_b"], pairs["relation"], strict=True)
    )
    assert counts.to_dict("records") == [
        {
            "candidate_pairs": len(candidates),
            "quantile": 0.95,
            "cut_cosine": pytest.approx(cut, abs=1e-12),
            "flagged_pairs": len(flagged),
            "known_pairs": 2,
            "known_min_cosine": pytest.approx(known["cosine"].min(), abs=1e-12),
            "known_median_cosine": pytest.approx(known["cosine"].median(), abs=1e-12),
            "reach_known_min": int((candidates["cosine"] >= known["cosine"].min()).sum()),
            "reach_known_median": int((candidates["cosine"] >= known["cosine"].median()).sum()),
        }
    ]


def test_embedding_pairs_never_pairs_problems_of_one_dev_fold_or_of_the_test_part():
    split = make_split()

    pairs, counts = embedding_pairs(make_vectors(split), split, quantile=0.0)  # every candidate

    flagged = pairs[pairs["kind"] == "flagged"]
    cross_fold = 3 * 3 * 3  # 3 pairs of dev folds, 3 x 3 pairs of problems each
    dev_test = 9 * 3
    assert counts.at[0, "candidate_pairs"] == len(flagged) == cross_fold + dev_test
    assert (flagged["relation"] == "dev_test").sum() == dev_test
    first, second = split.loc[flagged["sample_id_a"]], split.loc[flagged["sample_id_b"]]
    same_fold = first["fold"].to_numpy() == second["fold"].to_numpy()
    assert not same_fold.any()  # no pair inside a dev fold, nor two test problems (fold -1)


def test_the_pairs_file_stays_small(tmp_path):
    split = make_split()
    pairs, counts = embedding_pairs(make_vectors(split), split, quantile=0.0)
    path = tmp_path / PAIRS_FILE
    write_or_check(pairs, path)
    write_or_check(counts, tmp_path / PAIR_COUNTS_FILE)

    longest_row = max(len(line) for line in path.read_text(encoding="utf-8").splitlines())
    dataset_pairs = DATASET_PROBLEMS * (DATASET_PROBLEMS - 1) // 2
    dataset_rows = DATASET_KNOWN_PAIRS + (1 - PAIRS_QUANTILE) * dataset_pairs
    assert longest_row + 4 <= 64  # + 4: the dataset's two sample ids have 2 more digits each
    assert dataset_rows * 64 < 100_000  # bytes: about 1,300 rows


def test_embedding_pairs_refuses_vectors_of_other_problems_than_the_split():
    split = make_split()

    with pytest.raises(ValueError, match="differ on 1 problems"):
        embedding_pairs(make_vectors(split).drop(index=31), split)


@pytest.fixture(name="reference_parts")
def reference_parts_fixture(synthetic_raw, synthetic_labels, small_tfidf):
    """Return the synthetic dev (folds 0-3) and test (fold 4) problems, and the frozen F1s.

    The TF-IDF configuration is the conftest's ``small_tfidf`` (every word kept). The expected
    macro F1s come from the frozen TF-IDF's own code path, ``cross_fit`` of its
    ``build_pipeline`` and ``fit_tag_model``, on the raw problems, as the method ``tfidf`` is
    frozen.
    """
    dev_raw = synthetic_raw[synthetic_raw["fold"] < 4]
    test_raw = synthetic_raw[synthetic_raw["fold"] == 4]
    dev_labels = synthetic_labels.loc[dev_raw.index]
    out_of_fold = cross_fit(build_pipeline(small_tfidf), dev_raw, dev_labels, dev_raw["fold"])
    model = fit_tag_model(small_tfidf, dev_raw, dev_labels, dev_raw["fold"])
    return {
        "params": small_tfidf,
        "dev_raw": dev_raw,
        "test_raw": test_raw,
        "out_of_fold": out_of_fold,
        "model": model,
        "dev_f1": evaluate(dev_labels, out_of_fold.predictions).macro_f1,
        "test_f1": evaluate(synthetic_labels.loc[test_raw.index], model.predict(test_raw)).macro_f1,
    }


def test_tfidf_reference_re_derives_the_frozen_tfidf_rows(
    reference_parts, synthetic_labels, tmp_path
):
    parts = reference_parts

    rows = tfidf_reference(
        parts["dev_raw"],
        parts["test_raw"],
        synthetic_labels,
        parts["params"],
        expected_dev_macro_f1=parts["dev_f1"],
        expected_test_macro_f1=parts["test_f1"],
    )

    assert list(rows.columns) == ["part", "sample_id", "kind", *FOCUS_TAGS]
    expected = {
        ("dev", "score"): parts["out_of_fold"].scores,
        ("dev", "prediction"): parts["out_of_fold"].predictions,
        ("test", "score"): parts["model"].scores(parts["test_raw"]),
        ("test", "prediction"): parts["model"].predict(parts["test_raw"]),
    }
    assert list(dict.fromkeys(zip(rows["part"], rows["kind"], strict=True))) == list(expected)
    for (part, kind), frame in expected.items():
        selected = rows[(rows["part"] == part) & (rows["kind"] == kind)].set_index("sample_id")
        assert selected.index.tolist() == frame.index.tolist()
        np.testing.assert_array_equal(selected[list(FOCUS_TAGS)], frame.astype(float))
    path = tmp_path / TFIDF_REFERENCE_FILE
    assert write_or_check(rows, path).startswith("saved 90 rows")
    assert write_or_check(rows, path).endswith("holds the same 90 rows")


@pytest.mark.parametrize(
    ("shift", "what"),
    [
        ({"dev_f1": 1e-6}, "dev out-of-fold"),
        ({"test_f1": -1e-6}, "test"),
        ({"dev_f1": math.nan}, "dev out-of-fold"),  # NaN is never within the tolerance
    ],
)
def test_tfidf_reference_refuses_another_macro_f1_than_the_frozen_one(
    reference_parts, synthetic_labels, shift, what
):
    parts = reference_parts
    expected = {name: parts[name] + shift.get(name, 0.0) for name in ("dev_f1", "test_f1")}

    with pytest.raises(ValueError, match=f"reference's {what} macro F1 is"):
        tfidf_reference(
            parts["dev_raw"],
            parts["test_raw"],
            synthetic_labels,
            parts["params"],
            expected_dev_macro_f1=expected["dev_f1"],
            expected_test_macro_f1=expected["test_f1"],
        )
