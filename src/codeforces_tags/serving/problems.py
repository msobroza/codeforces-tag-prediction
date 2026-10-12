"""The Streamlit app's data: test-set problems, frozen rows, a typed problem, the answer per tag.

The app (``apps/streamlit_app.py``) holds layout only; the rules it applies live here and in
``serving.client``:

- ``load_test_set``: the test set of the challenge dataset, with each problem's true focus tags;
- ``frozen_prediction``: a frozen method's test row of one problem (``predictions.csv`` and, for
  a score method, ``scores.csv``), its focus tags only;
- ``display_text``: a text of a problem as the app shows it (``$$$x$$$`` becomes ``$x$``, and
  ``&lt;`` becomes ``<``), never as it is sent;
- ``problem_from_form``: a problem typed in the app's form, every text kept as typed;
- ``tag_table``: a model server's answer about one problem, focus tag by focus tag, beside the
  true focus tags and the frozen row;
- ``live_differences``: how a model server's answer differs from a frozen test row, by the rule
  of verify's level (a), which ``serving smoke`` applies too.

The test-set view is a demonstration on frozen test problems: it computes no metric and chooses
nothing (ADR 0001). This module imports neither mlflow nor torch.
"""

from __future__ import annotations

import html
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import (
    CLI_JSON_LIST_FIELDS,
    CLI_REQUIRED_FIELDS,
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import dataset_path, read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    SCORES_FILE,
    load_predictions,
    load_scores,
)
from codeforces_tags.provenance import check_dataset
from codeforces_tags.serving.client import ServedPrediction
from codeforces_tags.splitting import load_split, select_part

# The text fields of the app's form, in RAW_INPUT_COLUMNS order, with the label the form shows:
# the three parts of the statement, the notes and the solution. The sample inputs and outputs are
# typed as pairs, in a table (problem_from_form's sample_pairs).
FORM_FIELDS: dict[str, str] = {
    "prob_desc_description": "description",
    "prob_desc_input_spec": "input specification",
    "prob_desc_output_spec": "output specification",
    "prob_desc_notes": "notes",
    "source_code": "solution",
}


@dataclass(frozen=True)
class FrozenRow:
    """A frozen method's test row of one problem: what its ``freeze`` predicted, once.

    Attributes:
        tags: Each focus tag's frozen 0/1 prediction.
        scores: Each focus tag's frozen score, for a score method; None for a method without
            ``scores.csv``.
    """

    tags: dict[str, int]
    scores: dict[str, float] | None


def load_test_set(
    data_dir: str | os.PathLike[str] | None = None, split_path: Path = SPLIT_FILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the test set of the challenge dataset, with each problem's true focus tags.

    The dataset must be the challenge dataset, unchanged (``check_dataset``), as for the command
    line's ``evaluate --part test``: the frozen test rows were predicted on it.

    Args:
        data_dir: The dataset directory; None reads ``$CODEFORCES_DATA_DIR``, then the default.
        split_path: The frozen split.

    Returns:
        The raw test problems as ``select_part`` returns them (indexed by ``sample_id``, with
        their tags, fold and near-duplicate group), and their true focus tags as ``focus_labels``
        gives them (0/1, one column per focus tag).

    Raises:
        CliError: If the directory is missing or is not the challenge dataset, or if the split
            cannot be read or holds other problems; the message names the directory.
    """
    path = dataset_path(data_dir)
    try:
        check_dataset(path)
        problems = select_part(read_samples(path), load_split(split_path), "test")
    except (OSError, ValueError) as error:
        raise CliError(f"the test set of {path} cannot be read: {error}") from error
    return problems, focus_labels(problems)


def frozen_prediction(
    name: str, sample_id: int, assets_dir: Path = METHODS_ASSETS_DIR
) -> FrozenRow | None:
    """Read a frozen method's test row of one problem, its focus tags only.

    Args:
        name: The method's registry key.
        sample_id: The problem.
        assets_dir: The folder of the frozen methods.

    Returns:
        The frozen focus tags of the problem, and their scores for a score method; None when
        the method is not frozen in ``assets_dir`` or has no test row for the problem.

    Raises:
        ValueError: As ``load_predictions`` and ``load_scores``: a frozen file that is not
            indexed by part and sample id, or a ``predictions.csv`` that holds values other than
            0 and 1.
    """
    directory = Path(assets_dir) / name
    key = ("test", int(sample_id))
    if not (directory / PREDICTIONS_FILE).is_file():
        return None
    predictions = load_predictions(directory)
    if key not in predictions.index:
        return None
    scores = None
    if (directory / SCORES_FILE).is_file():
        scores = _focus_row(load_scores(directory), key, float)
    return FrozenRow(tags=_focus_row(predictions, key, int), scores=scores)


def display_text(text: str | None) -> str:
    """Write a text of a problem as the app shows it, never as it is sent.

    Codeforces writes inline math between ``$$$`` (display math between six dollars), and
    Streamlit's Markdown reads it between ``$`` (and ``$$``). The dataset also keeps the HTML
    entities of the scraped page (``&lt;``, ``&gt;``, ``&amp;``), inside formulas too, where
    KaTeX would show them as errors. The model server reads the text as the dataset stores it,
    entities and ``$$$`` included: only the display changes.

    The math delimiters are rewritten first, then the entities are decoded once, so a ``$``
    that an entity decodes to never joins its neighbours into a delimiter, and ``&amp;lt;``
    shows as the ``&lt;`` that it spells.

    Args:
        text: A field of a problem; None, or NaN for a field the problem lacks, shows nothing.

    Returns:
        The text with each ``$$$`` written ``$`` and each HTML entity decoded; ``""`` for a
        missing field.
    """
    if not isinstance(text, str):
        return ""
    return html.unescape(text.replace("$$$", "$"))


def tag_table(
    prediction: ServedPrediction,
    truth: Mapping[str, int] | None = None,
    frozen: FrozenRow | None = None,
) -> pd.DataFrame:
    """Lay out a model server's answer about one problem, focus tag by focus tag.

    Args:
        prediction: The model server's answer about the problem.
        truth: The problem's true focus tags (0/1 by tag), for a test-set problem; None for a
            typed one.
        frozen: The served method's frozen test row of the problem, if it has one.

    Returns:
        One row per focus tag, in ``FOCUS_TAGS`` order, indexed by ``tag``: ``predicted``
        (0/1); ``score`` and ``threshold`` (floats: NaN for a method without scores, the per-tag
        threshold ``inf`` for a tag the method never predicts); ``true`` and ``frozen`` (nullable
        0/1, NA without a truth or a frozen row); and ``agrees`` (whether ``predicted`` equals
        ``frozen``, NA without a frozen row).
    """
    table = pd.DataFrame(
        {
            "predicted": [int(tag in prediction.tags) for tag in FOCUS_TAGS],
            "score": _floats(prediction.scores),
            "threshold": _floats(prediction.thresholds),
            "true": _zeros_ones(truth),
            "frozen": _zeros_ones(None if frozen is None else frozen.tags),
        },
        index=pd.Index(FOCUS_TAGS, name="tag"),
    )
    table["agrees"] = (table["predicted"] == table["frozen"]).astype("boolean")
    return table


def problem_from_form(
    texts: Mapping[str, str],
    sample_pairs: Sequence[tuple[str, str]],
    sample_id: int = 0,
) -> pd.DataFrame:
    """Build the raw problem typed in the app's form, as the model server reads it.

    Every text is kept as typed, character for character (math between ``$$$``, any letter,
    Windows line ends, leading and trailing white space): only ``display_text`` rewrites a
    text, and only for display.

    Args:
        texts: The typed text of each field of ``FORM_FIELDS``; a field left out counts as
            blank.
        sample_pairs: The (sample input, sample output) rows of the form's table, in order. A
            blank side (empty, white space only, or a cell left empty: None or NaN) is sent as
            ``""``, and a row blank on both sides is dropped.
        sample_id: The ``sample_id`` of the problem.

    Returns:
        One row indexed by ``sample_id``, with the columns of ``RAW_INPUT_COLUMNS`` in order:
        each typed text as typed, None for a blank optional text, and the sample inputs and
        outputs as lists of texts (empty without a row).

    Raises:
        ValueError: If ``texts`` holds a field that ``FORM_FIELDS`` does not, or if the
            description or the solution is blank; the message names the form's labels.
    """
    _check_form(texts)
    row = {field: _typed(texts.get(field)) for field in FORM_FIELDS}
    pairs = _kept_pairs(sample_pairs)
    inputs, outputs = CLI_JSON_LIST_FIELDS
    row[inputs] = [given for given, _ in pairs]
    row[outputs] = [expected for _, expected in pairs]
    return pd.DataFrame(
        [[row[column] for column in RAW_INPUT_COLUMNS]],
        index=pd.Index([sample_id], name="sample_id"),
        columns=list(RAW_INPUT_COLUMNS),
    )


def live_differences(
    prediction: ServedPrediction, frozen: FrozenRow, *, score_atol: float
) -> list[str]:
    """Say how a model server's answer about a test-set problem differs from its frozen row.

    The rule of verify's level (a) (CLAUDE.md, Adding a tagging method, step 7): each score
    within ``max(score_atol, TUNING_EXPORT_TOLERANCE)`` of its frozen score (rtol 0), and a 0/1
    flip allowed only where the frozen score lies within ``score_atol`` of the per-tag threshold
    the model server answers; a method without scores must give its frozen focus tags exactly.

    Args:
        prediction: The model server's answer about the problem.
        frozen: The served method's frozen test row of the problem.
        score_atol: The method's ``MethodSpec.score_atol``.

    Returns:
        One line per difference, naming its focus tag; empty when the answer matches.
    """
    if frozen.scores is None:
        return [
            _flip(tag, prediction, frozen)
            for tag in FOCUS_TAGS
            if _flipped(tag, prediction, frozen)
        ]
    lacking = _without_scores(prediction)
    if lacking:
        return [f"the answer holds no score or per-tag threshold of {', '.join(lacking)}"]
    lines = []
    for tag in FOCUS_TAGS:
        lines.extend(_score_differences(tag, prediction, frozen, score_atol))
    return lines


def _check_form(texts: Mapping[str, str]) -> None:
    """Refuse a field the form does not have, and a blank description or solution."""
    unknown = sorted(set(texts) - set(FORM_FIELDS))
    if unknown:
        raise ValueError(f"the form has no field {unknown}: its fields are {list(FORM_FIELDS)}.")
    empty = [FORM_FIELDS[field] for field in CLI_REQUIRED_FIELDS if _blank(texts.get(field))]
    if empty:
        named = " and ".join(f"the {label} is empty" for label in empty)
        raise ValueError(f"{named}: a problem needs a description and a solution.")


def _focus_row(frame: pd.DataFrame, key: tuple[str, int], kind: type) -> dict:
    """Return the value of each focus tag in one row of a frozen file, converted by ``kind``."""
    row = frame.loc[key]
    return {tag: kind(row[tag]) for tag in FOCUS_TAGS}


def _blank(value: object) -> bool:
    """Tell whether a typed value is blank: not a text (None, NaN), or white space only."""
    return not isinstance(value, str) or not value.strip()


def _typed(value: object) -> str | None:
    """Return a typed text as it is, or None when it is blank."""
    return None if _blank(value) else value


def _kept_pairs(sample_pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop the sample pairs blank on both sides, and write a blank side of the others ``""``."""
    return [
        (_side(given), _side(expected))
        for given, expected in sample_pairs
        if not (_blank(given) and _blank(expected))
    ]


def _side(value: object) -> str:
    """Return one side of a sample pair: the text as typed, or ``""`` when it is blank."""
    return "" if _blank(value) else value


def _floats(values: Mapping[str, float]) -> list[float]:
    """Return the float of each focus tag, in order; NaN for a tag without one."""
    return [float(values.get(tag, math.nan)) for tag in FOCUS_TAGS]


def _zeros_ones(values: Mapping[str, int] | None) -> pd.arrays.IntegerArray:
    """Return the 0/1 of each focus tag as nullable integers, in order; all NA without values."""
    data = [pd.NA] * len(FOCUS_TAGS) if values is None else [values[tag] for tag in FOCUS_TAGS]
    return pd.array(data, dtype="Int64")


def _without_scores(prediction: ServedPrediction) -> list[str]:
    """Return the focus tags whose score or per-tag threshold a model server's answer lacks."""
    return [
        tag
        for tag in FOCUS_TAGS
        if tag not in prediction.scores or tag not in prediction.thresholds
    ]


def _served(tag: str, prediction: ServedPrediction) -> int:
    """Return the 0/1 that a model server predicted for a focus tag."""
    return int(tag in prediction.tags)


def _flipped(tag: str, prediction: ServedPrediction, frozen: FrozenRow) -> bool:
    """Tell whether a model server's 0/1 for a focus tag differs from the frozen one."""
    return _served(tag, prediction) != frozen.tags[tag]


def _flip(tag: str, prediction: ServedPrediction, frozen: FrozenRow) -> str:
    """Name a focus tag whose 0/1 differs: what the model server and the frozen row predict."""
    return f"{tag}: the model server predicts {_served(tag, prediction)}, frozen {frozen.tags[tag]}"


def _score_differences(
    tag: str, prediction: ServedPrediction, frozen: FrozenRow, score_atol: float
) -> list[str]:
    """Compare a score method's answer for one focus tag with its frozen row (verify's (a))."""
    score, frozen_score = prediction.scores[tag], frozen.scores[tag]
    threshold = prediction.thresholds[tag]
    atol = max(score_atol, TUNING_EXPORT_TOLERANCE)
    lines = []
    if not abs(score - frozen_score) <= atol:  # true for a NaN score too
        lines.append(
            f"{tag}: score {score:.10g}, frozen {frozen_score:.10g}: further apart than {atol:g}"
        )
    if _flipped(tag, prediction, frozen) and not abs(frozen_score - threshold) <= score_atol:
        lines.append(
            f"{_flip(tag, prediction, frozen)}, and its frozen score {frozen_score:.10g} lies "
            f"further than {score_atol:g} from the per-tag threshold {threshold:.10g}"
        )
    return lines
