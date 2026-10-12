"""The shared analysis files of the f2llm methods, in ``assets/f2llm/``.

They sit outside ``assets/methods/``, where CI checks every folder as a frozen method. Each is
written once, then checked on later runs (``write_or_check``: floats within
``TUNING_EXPORT_TOLERANCE``, every other value exactly).

- ``input_lengths.csv`` (written by ``embed``): the tokens of each problem's two encoder inputs,
  end-of-sequence token included, untruncated.
- ``embedding_pairs.csv`` and ``embedding_pairs_counts.csv`` (written by ``embed``): the
  near-duplicate check on the statement embeddings. The pairs file holds every pair inside a
  frozen near-duplicate group (``known``), and the pairs from different groups, in different
  dev folds or in different parts, whose cosine reaches the ``PAIRS_QUANTILE`` quantile of all
  such pairs (``flagged``); the counts file holds one row of counts, among them the pairs that
  reach the lowest and the median known-pair cosine.
- ``tfidf_reference.csv`` (written by ``reference``): the frozen TF-IDF configuration's dev
  out-of-fold and test rows, scores and 0/1 predictions, the same rows as the frozen ``tfidf``
  method's ``scores.csv`` and ``predictions.csv``. It re-derives TF-IDF's single test score, and
  scores nothing new.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from codeforces_tags.config import TUNING_BLAS_THREADS, TUNING_EXPORT_TOLERANCE
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.f2llm.embedding import INPUTS, encoder_inputs
from codeforces_tags.methods.f2llm.encoder import EncoderConfig, token_counts
from codeforces_tags.methods.f2llm.estimators import make_raw_tfidf
from codeforces_tags.methods.f2llm.settings import PAIRS_QUANTILE
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds, apply_thresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems

INPUT_LENGTHS_FILE = "input_lengths.csv"
PAIRS_FILE = "embedding_pairs.csv"
PAIR_COUNTS_FILE = "embedding_pairs_counts.csv"
TFIDF_REFERENCE_FILE = "tfidf_reference.csv"
# The kinds of pairs in PAIRS_FILE, and how the two problems of a pair sit in the split.
KNOWN, FLAGGED = "known", "flagged"
SAME_GROUP, CROSS_FOLD, DEV_TEST = "same_group", "cross_fold", "dev_test"


def write_or_check(
    frame: pd.DataFrame, path: Path, *, atol: float = TUNING_EXPORT_TOLERANCE
) -> str:
    """Write a table once; later, check that it holds the same rows.

    The first call writes the CSV under a temporary name, then renames it. A later call
    writes nothing: the stored table and ``frame``, both read as their CSV reads, must have the
    same columns, in order, and the same number of rows; float columns must agree within
    ``atol`` (rtol 0, NaN equal to NaN), every other column exactly.

    Args:
        frame: The table, written without its index.
        path: The CSV file; its folder is created if missing.
        atol: The tolerance of the float columns.

    Returns:
        A line to print: the file was saved, or it holds the same rows.

    Raises:
        ValueError: If the stored table differs from ``frame``.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.partial")
        frame.to_csv(partial, index=False)
        os.replace(partial, path)
        return f"saved {len(frame):,} rows to {path}"
    stored = pd.read_csv(path, float_precision="round_trip")
    if not _same_table(stored, _as_read(frame), atol):
        raise ValueError(
            f"{path} differs from the recomputed table (floats by more than {atol}, or another "
            "value): the code, data, split or encoder changed since it was written. Delete it "
            "only if it was never committed."
        )
    return f"{path} holds the same {len(frame):,} rows"


def input_lengths(
    raw_problems: pd.DataFrame, tokenizer: object, config: EncoderConfig
) -> pd.DataFrame:
    """Count the tokens of each problem's encoder inputs, untruncated.

    Args:
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.
        tokenizer: The encoder's tokenizer (``load_tokenizer``).
        config: The encoder configuration, which holds the instructions.

    Returns:
        ``sample_id``, ``statement_tokens`` and ``solution_tokens`` (int64, end-of-sequence
        token included), one row per problem in row order: the rows of ``INPUT_LENGTHS_FILE``.
    """
    inputs = encoder_inputs(raw_to_problems(raw_problems), config)
    counts = {f"{name}_tokens": token_counts(inputs[name], tokenizer) for name in INPUTS}
    return pd.DataFrame({"sample_id": raw_problems.index.to_numpy(dtype=np.int64), **counts})


def embedding_pairs(
    statement_vectors: pd.DataFrame, split: pd.DataFrame, *, quantile: float = PAIRS_QUANTILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the pairs of problems whose statement embeddings are near-duplicates.

    The candidates are the pairs from different near-duplicate groups that sit in different
    dev folds (``cross_fold``) or in different parts (``dev_test``): the pairs that would leak
    if they were the same problem. A candidate is flagged when its cosine reaches the
    ``quantile`` quantile of every candidate's cosine. Pairs inside a group are listed as
    ``known``, with relation ``same_group``.

    Args:
        statement_vectors: One row per problem of ``split``, in any order, indexed by
            ``sample_id``, one column per dimension of the statement embedding.
        split: The frozen split (``load_split()``): ``part``, ``fold`` and ``group``, indexed
            by ``sample_id``.
        quantile: The share of candidates below the cut.

    Returns:
        ``(pairs, counts)``. ``pairs``: ``sample_id_a`` < ``sample_id_b``, ``cosine``,
        ``kind`` (``known`` or ``flagged``) and ``relation``, the known pairs then the
        flagged ones, each sorted by sample ids. ``counts``: one row with ``candidate_pairs``,
        ``quantile``, ``cut_cosine``, ``flagged_pairs``, ``known_pairs``,
        ``known_min_cosine``, ``known_median_cosine``, and ``reach_known_min`` and
        ``reach_known_median``, the candidates whose cosine reaches those two.

    Raises:
        ValueError: If the vectors and the split hold other problems.
    """
    vectors = statement_vectors.sort_index()
    split = _aligned_split(vectors.index, split)
    cosines = _cosines(vectors.to_numpy(dtype=np.float64))
    known, candidates = _pair_masks(split)
    candidate_cosines = cosines[candidates]
    cut = float(np.quantile(candidate_cosines, quantile))
    flagged = candidates & (cosines >= cut)
    ids = vectors.index.to_numpy(dtype=np.int64)
    is_dev = (split["part"] == "dev").to_numpy()
    pairs = pd.concat(
        [
            _pair_rows(mask, cosines, ids, is_dev, kind)
            for mask, kind in ((known, KNOWN), (flagged, FLAGGED))
        ],
        ignore_index=True,
    )
    return pairs, _pair_counts(candidate_cosines, cosines[known], quantile, cut)


def tfidf_reference(  # pylint: disable=too-many-arguments
    dev_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    labels: pd.DataFrame,
    params: ModelParams,
    *,
    expected_dev_macro_f1: float,
    expected_test_macro_f1: float,
) -> pd.DataFrame:
    """Re-derive the frozen TF-IDF configuration's dev out-of-fold and test rows.

    ``CrossFittedTagThresholds(make_raw_tfidf(params), refit=True)`` is fitted on the dev
    problems with their folds: its out-of-fold scores and cross-fitted predictions are the dev
    rows; the refit's scores, and its predictions at the final thresholds, are the test rows.

    Args:
        dev_raw: The raw dev problems with their ``fold`` (``select_part(raw, split, "dev")``).
        test_raw: The raw test problems.
        labels: 0/1 focus labels of every problem of both parts, one column per tag.
        params: The frozen TF-IDF configuration (``tfidf_params()``).
        expected_dev_macro_f1: The frozen cross-fitted out-of-fold macro F1.
        expected_test_macro_f1: The frozen test macro F1.

    Returns:
        ``part``, ``sample_id``, ``kind`` (``score`` or ``prediction``), then one float column
        per tag: dev scores, dev predictions, test scores, test predictions, each in the row
        order of its part. The rows of ``TFIDF_REFERENCE_FILE``.

    Raises:
        ValueError: If either macro F1 differs from its expected value by more than
            ``TUNING_EXPORT_TOLERANCE``: the reference is then not the frozen TF-IDF's.
    """
    dev_labels, test_labels = labels.loc[dev_raw.index], labels.loc[test_raw.index]
    model = CrossFittedTagThresholds(make_raw_tfidf(params), refit=True)
    with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
        model.fit(dev_raw, dev_labels, groups=dev_raw["fold"])
        test_scores = model.decision_function(test_raw)
    tags = labels.columns
    rows = {
        ("dev", "score"): pd.DataFrame(model.oof_scores_, index=dev_raw.index, columns=tags),
        ("dev", "prediction"): pd.DataFrame(
            model.oof_predictions_, index=dev_raw.index, columns=tags
        ),
        ("test", "score"): pd.DataFrame(test_scores, index=test_raw.index, columns=tags),
    }
    thresholds = pd.Series(model.thresholds_, index=tags)
    rows["test", "prediction"] = apply_thresholds(rows["test", "score"], thresholds)
    _check_macro_f1("dev out-of-fold", dev_labels, rows["dev", "prediction"], expected_dev_macro_f1)
    _check_macro_f1("test", test_labels, rows["test", "prediction"], expected_test_macro_f1)
    return _reference_rows(rows)


def _as_read(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` as reading its CSV gives it, with the dtypes of a stored table."""
    text = frame.to_csv(index=False)
    return pd.read_csv(io.StringIO(text), float_precision="round_trip")


def _same_table(stored: pd.DataFrame, new: pd.DataFrame, atol: float) -> bool:
    """Tell whether two tables have the same columns and rows, floats within ``atol``."""
    if list(stored.columns) != list(new.columns) or len(stored) != len(new):
        return False
    return all(_same_column(stored[column], new[column], atol) for column in stored.columns)


def _same_column(stored: pd.Series, new: pd.Series, atol: float) -> bool:
    """Compare two float columns within ``atol`` (NaN equal to NaN), any other exactly."""
    if pd.api.types.is_float_dtype(stored) and pd.api.types.is_float_dtype(new):
        return bool(np.allclose(stored, new, rtol=0, atol=atol, equal_nan=True))
    return stored.equals(new)


def _aligned_split(sample_ids: pd.Index, split: pd.DataFrame) -> pd.DataFrame:
    """Return the split's rows in the order of ``sample_ids``, which must be its problems."""
    differ = split.index.symmetric_difference(sample_ids)
    if len(differ):
        raise ValueError(
            f"The statement vectors and the split differ on {len(differ)} problems: embed every "
            "problem of the split, as read_samples() returns them."
        )
    return split.loc[sample_ids]


def _cosines(vectors: np.ndarray) -> np.ndarray:
    """Return the cosine of every pair of rows, an (n, n) float64 matrix."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return unit @ unit.T


def _pair_masks(split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n, n) masks of the known pairs and of the candidates, each pair once."""
    groups = split["group"].to_numpy()
    folds = split["fold"].to_numpy()
    is_dev = (split["part"] == "dev").to_numpy()
    upper = np.triu(np.ones((len(split), len(split)), dtype=bool), k=1)
    same_group = groups[:, None] == groups[None, :]
    dev_test = is_dev[:, None] != is_dev[None, :]
    cross_fold = is_dev[:, None] & is_dev[None, :] & (folds[:, None] != folds[None, :])
    return upper & same_group, upper & ~same_group & (dev_test | cross_fold)


def _pair_rows(
    mask: np.ndarray, cosines: np.ndarray, ids: np.ndarray, is_dev: np.ndarray, kind: str
) -> pd.DataFrame:
    """List the pairs of a mask, sorted by sample ids, with their cosine, kind and relation."""
    first, second = np.nonzero(mask)
    if kind == KNOWN:
        relation = np.full(len(first), SAME_GROUP)
    else:
        relation = np.where(is_dev[first] != is_dev[second], DEV_TEST, CROSS_FOLD)
    return pd.DataFrame(
        {
            "sample_id_a": ids[first],
            "sample_id_b": ids[second],
            "cosine": cosines[first, second],
            "kind": kind,
            "relation": relation,
        }
    )


def _pair_counts(
    candidate_cosines: np.ndarray, known_cosines: np.ndarray, quantile: float, cut: float
) -> pd.DataFrame:
    """Count the candidates at the quantile cut and at the lowest and median known cosine."""
    lowest, median = float(known_cosines.min()), float(np.median(known_cosines))
    counts = {
        "candidate_pairs": candidate_cosines.size,
        "quantile": quantile,
        "cut_cosine": cut,
        "flagged_pairs": int((candidate_cosines >= cut).sum()),
        "known_pairs": known_cosines.size,
        "known_min_cosine": lowest,
        "known_median_cosine": median,
        "reach_known_min": int((candidate_cosines >= lowest).sum()),
        "reach_known_median": int((candidate_cosines >= median).sum()),
    }
    return pd.DataFrame([counts])


def _check_macro_f1(
    what: str, truth: pd.DataFrame, predictions: pd.DataFrame, expected: float
) -> None:
    """Fail unless the predictions' macro F1 is ``expected``, within the export tolerance."""
    macro_f1 = evaluate(truth, predictions).macro_f1
    if not abs(macro_f1 - expected) <= TUNING_EXPORT_TOLERANCE:  # not >: NaN is never within it
        raise ValueError(
            f"The TF-IDF reference's {what} macro F1 is {macro_f1!r}, not the frozen "
            f"{expected!r} (tolerance {TUNING_EXPORT_TOLERANCE}): the code, data or split "
            "differ from the frozen TF-IDF's, so nothing is written."
        )


def _reference_rows(rows: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Stack the reference rows under their ``part`` and ``kind``, tags as float columns."""
    frames = [
        frame.astype(float).rename_axis("sample_id").reset_index().assign(part=part, kind=kind)
        for (part, kind), frame in rows.items()
    ]
    stacked = pd.concat(frames, ignore_index=True)
    return stacked[["part", "sample_id", "kind", *rows["dev", "score"].columns]]
