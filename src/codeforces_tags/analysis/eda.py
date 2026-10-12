"""Exploratory data analysis: statistics that describe the dataset and relate inputs to tags.

The analysis notebooks call these functions instead of defining helpers inline, so every
statistic they show is defined once, documented and tested.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer

from codeforces_tags.config import (
    FOCUS_TAGS,
    TERMS_MIN_DF,
    TERMS_MIN_ODDS_RATIO,
    TERMS_TOP_K,
    TEXT_FIELDS,
)
from codeforces_tags.preprocessing.text import (
    GLUED_SENTENCE,
    cyrillic_share,
    is_russian,
    strip_latex,
)

_CYRILLIC_WORD = re.compile(r"\w*[А-Яа-яЁё]\w*")
# A notes heading glued to the next word ("NoteIn the first...", "Note12 = 4 + 8"). A sentence
# such as "Note that..." keeps its space, so it is not counted.
_GLUED_NOTES_HEADING = re.compile(r"\s*(?:Note|Примечание)(?=[A-ZА-ЯЁ$\d])")
# Letter runs, digit runs and any other single character: a rough proxy for subword tokens.
_TOKEN_PIECE = re.compile(r"[^\W\d_]+|\d+|\S")

# Dataset profile -----------------------------------------------------------------------------


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise every column of a frame.

    Args:
        frame: Samples, raw or preprocessed.

    Returns:
        One row per column with its value types, missing and empty counts, number of unique
        values and an example (the first non-empty value). Lists count as one value whatever
        their order.
    """
    rows = {}
    for name, values in frame.items():
        present = values.dropna()
        hashable = present.map(lambda v: tuple(sorted(set(v))) if isinstance(v, list) else v)
        example = next((v for v in present if not _is_empty(v)), next(iter(present), None))
        rows[name] = {
            "value type": ", ".join(sorted({type(v).__name__ for v in present})),
            "missing": values.isna().sum(),
            "empty": present.map(_is_empty).sum(),
            "unique": hashable.nunique(),
            "example": repr(example)[:60],
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("variable")


def missing_value_counts(
    frame: pd.DataFrame, sentinels: dict[str, object] | None = None
) -> pd.DataFrame:
    """Count the three kinds of missing value in each column.

    Args:
        frame: Raw samples.
        sentinels: Placeholder meaning "unknown", per column, e.g. ``{"difficulty": -1}``.

    Returns:
        Columns ``None``, ``empty string / list`` and ``sentinel``, for the columns that have at
        least one missing value, sorted by total.
    """
    counts = pd.DataFrame(
        {
            "None": frame.isna().sum(),
            "empty string / list": frame.apply(lambda column: column.map(_is_empty).sum()),
            "sentinel": 0,
        }
    )
    for column, sentinel in (sentinels or {}).items():
        counts.loc[column, "sentinel"] = (frame[column] == sentinel).sum()
    counts = counts[counts.sum(axis=1) > 0]
    return counts.loc[counts.sum(axis=1).sort_values().index]


def scraping_artefact_shares(frame: pd.DataFrame) -> pd.Series:
    """Measure the scraping artefacts that preprocessing fixes.

    Args:
        frame: Samples with description, notes and tags, raw or preprocessed.

    Returns:
        The share of statements with a glued paragraph break, of notes whose heading is glued
        to the text, and of problems with a repeated tag.
    """
    notes = frame["prob_desc_notes"].dropna()
    return pd.Series(
        {
            "statements with a glued paragraph break": frame["prob_desc_description"]
            .map(lambda text: GLUED_SENTENCE.search(text) is not None)
            .mean(),
            "notes with a heading glued to the text": notes.map(
                lambda text: _GLUED_NOTES_HEADING.match(text) is not None
            ).mean(),
            "problems with a repeated tag": frame["tags"]
            .map(lambda tags: len(tags) != len(set(tags)))
            .mean(),
        }
    )


def cyrillic_field_counts(frame: pd.DataFrame, fields: Sequence[str] = TEXT_FIELDS) -> pd.DataFrame:
    """Count, per text field, the texts with Cyrillic letters and the texts in Russian.

    Args:
        frame: Samples with the given text fields.
        fields: Text columns to inspect.

    Returns:
        One row per field: texts with any Cyrillic letter, and texts in Russian (``is_russian``).
    """
    rows = {}
    for field in fields:
        texts = frame[field].dropna()
        rows[field] = {
            "any Cyrillic letter": (texts.map(cyrillic_share) > 0).sum(),
            "mostly Cyrillic (Russian)": texts.map(is_russian).sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def cyrillic_words(text: str) -> list[str]:
    """Return the distinct words of ``text`` that contain a Cyrillic letter, sorted."""
    return sorted(set(_CYRILLIC_WORD.findall(text)))


def near_duplicate_summary(
    groups: pd.Series, tags: pd.Series, focus_tags: Sequence[str] = FOCUS_TAGS
) -> pd.Series:
    """Describe the groups of near-duplicate problems.

    Args:
        groups: Group id per problem, from ``near_duplicate_groups``.
        tags: Tag list per problem, with the same index.
        focus_tags: Tags whose agreement within a group is also reported.

    Returns:
        The number of groups with several problems, the largest group size, the number of
        problems in such groups, and how many of these groups agree on all tags and on the
        focus tags.
    """
    size = groups.map(groups.value_counts())
    grouped = size > 1
    group_ids = groups[grouped]
    tag_sets = tags[grouped].map(frozenset)
    focus_sets = tag_sets.map(lambda tag_set: tag_set & frozenset(focus_tags))
    return pd.Series(
        {
            "groups of near-duplicates": group_ids.nunique(),
            "largest group size": size.max(),
            "problems in a group": grouped.sum(),
            "groups with identical tags": (tag_sets.groupby(group_ids).nunique() == 1).sum(),
            "groups with identical focus tags": (
                focus_sets.groupby(group_ids).nunique() == 1
            ).sum(),
        }
    )


# Labels --------------------------------------------------------------------------------------


def label_counts(labels: pd.Series) -> pd.Series:
    """Count the rows carrying each label.

    Args:
        labels: One collection per row, e.g. tags or imported modules. A label repeated within
            a row counts once.

    Returns:
        Number of rows per label, most frequent first.
    """
    return labels.map(set).explode().dropna().value_counts().rename("problems")


def cooccurrence(
    rows: pd.DataFrame, columns: pd.DataFrame | None = None, normalize: bool = False
) -> pd.DataFrame:
    """Count how often two labels appear on the same problem.

    Args:
        rows: Indicator matrix from ``multi_hot``; its labels become the rows.
        columns: Second indicator matrix with the same index. Defaults to ``rows``.
        normalize: If True, return P(column label | row label) instead of counts: the share
            of problems with the row label that also have the column label.

    Returns:
        A rows x columns matrix of counts, or of conditional probabilities.
    """
    columns = rows if columns is None else columns
    counts = rows.T @ columns
    return counts.div(rows.sum(), axis=0) if normalize else counts


def lift(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Measure how much each row label raises the chance of each column label.

    Args:
        rows: Indicator matrix of the conditioning labels, e.g. imported modules.
        labels: Indicator matrix of the target labels, e.g. focus tags, with the same index.

    Returns:
        P(label | row label) / P(label). Above 1, the row label makes the label more likely.
    """
    return cooccurrence(rows, labels, normalize=True) / labels.mean()


def summarise_by_label(
    values: pd.DataFrame, indicators: pd.DataFrame, statistic: str = "mean"
) -> pd.DataFrame:
    """Aggregate each column of ``values`` over the rows that carry each label.

    Args:
        values: One row per problem, e.g. 0/1 cue flags or resource limits.
        indicators: 0/1 label matrix with the same index, e.g. from ``multi_hot``.
        statistic: A pandas reduction name, such as ``"mean"`` or ``"median"``.

    Returns:
        One row per column of ``values`` and one column per label, plus ``"all problems"``.
    """
    summary = pd.DataFrame(
        {label: values[has_label == 1].agg(statistic) for label, has_label in indicators.items()}
    )
    summary["all problems"] = values.agg(statistic)
    return summary


def values_by_label(
    values: pd.Series, indicators: pd.DataFrame, none_label: str = "(none)"
) -> pd.DataFrame:
    """Reshape values into one row per (label, value) pair, e.g. for a boxplot per tag.

    Args:
        values: One value per row of ``indicators``.
        indicators: 0/1 label matrix, e.g. from ``multi_hot``.
        none_label: Label given to rows that have no label at all.

    Returns:
        Columns ``label`` and ``value``. A row with several labels appears once per label.
        Missing values are dropped.
    """
    pairs = indicators.stack()
    pairs = pairs[pairs == 1].index
    labelled = pd.DataFrame(
        {
            "label": pairs.get_level_values(-1),
            "value": values.reindex(pairs.get_level_values(0)).to_numpy(),
        }
    )
    unlabelled = pd.DataFrame(
        {"label": none_label, "value": values[indicators.sum(axis=1) == 0].to_numpy()}
    )
    return pd.concat([labelled, unlabelled], ignore_index=True).dropna(subset=["value"])


# Text and code -------------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Count the words of a text outside its math spans."""
    return len(strip_latex(text).split())


def approximate_token_count(text: str) -> int:
    """Count the words, numbers and other symbols of a text.

    Subword tokenizers split words further, so a real token count is higher.

    Args:
        text: Any text, LaTeX included.

    Returns:
        The number of pieces, a lower-bound proxy for the number of tokens.
    """
    return len(_TOKEN_PIECE.findall(text))


def code_line_count(source_code: str) -> int:
    """Count the non-blank lines of a solution."""
    return sum(1 for line in source_code.splitlines() if line.strip())


def parses_as_python3(source_code: str) -> bool:
    """Tell whether Python 3's ``ast`` module can parse a solution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # Python 2 string escapes
        try:
            ast.parse(source_code)
        except (SyntaxError, ValueError):
            return False
    return True


# Associations --------------------------------------------------------------------------------


def odds_ratio(exposed: pd.Series, outcome: pd.Series, strata: pd.Series | None = None) -> float:
    """Compute the odds ratio of an outcome given an exposure.

    Args:
        exposed: Exposure per problem, as booleans or 0/1.
        outcome: Outcome per problem, as booleans or 0/1.
        strata: Optional group per problem. When given, the Mantel-Haenszel odds ratio pools
            the within-group comparisons, which removes the effect of the grouping variable.
            Rows whose group is missing are ignored.

    Returns:
        The crude odds ratio, or the Mantel-Haenszel odds ratio when ``strata`` is given.
        ``inf`` when no pair supports the reverse association, ``nan`` when undefined.
    """
    frame = pd.DataFrame({"exposed": exposed.astype(bool), "outcome": outcome.astype(bool)})
    strata = pd.Series(0, index=frame.index) if strata is None else strata
    numerator = denominator = 0.0
    for _, group in frame.groupby(strata):
        exp, out = group["exposed"], group["outcome"]
        a = (exp & out).sum()  # exposed, with outcome
        b = (exp & ~out).sum()  # exposed, without outcome
        c = (~exp & out).sum()  # not exposed, with outcome
        d = (~exp & ~out).sum()  # not exposed, without outcome
        numerator += a * d / len(group)
        denominator += b * c / len(group)
    if denominator == 0:
        return np.inf if numerator else np.nan
    return float(numerator / denominator)


def distinctive_terms(
    texts: pd.Series,
    labels: pd.DataFrame,
    top_k: int = TERMS_TOP_K,
    min_df: int = TERMS_MIN_DF,
    min_odds_ratio: float = TERMS_MIN_ODDS_RATIO,
    **vectorizer_args: Any,
) -> pd.DataFrame:
    """Find the terms that best separate documents with a label from those without.

    For each label and each term, a 2x2 table "term present x label present" gives an odds
    ratio (OR). Terms are ranked by the z-score of log(OR): dividing by the standard error
    stops a term seen in a handful of documents from winning on chance. The z-score also
    grows with support, which favours generic words for large labels, so terms with an OR
    below ``min_odds_ratio`` are dropped first.

    Args:
        texts: One document per row.
        labels: 0/1 label columns aligned on ``texts``, e.g. from ``multi_hot``.
        top_k: Number of terms to return per label.
        min_df: Ignore terms found in fewer documents than this.
        min_odds_ratio: Ignore terms whose odds ratio is below this.
        **vectorizer_args: Passed to ``CountVectorizer`` (e.g. ``token_pattern``).

    Returns:
        One column per label and ``top_k`` rows of ``"term (OR x.x)"``, best first; ``""``
        where fewer terms qualify.
    """
    vectorizer = CountVectorizer(binary=True, min_df=min_df, **vectorizer_args)
    present = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()

    columns = {}
    for label, has_label in labels.items():
        ratio, z = _odds_ratio_and_z(present, has_label.to_numpy().astype(bool))
        z[ratio < min_odds_ratio] = -np.inf
        terms = [
            f"{vocabulary[i]} (OR {ratio[i]:.1f})" if np.isfinite(z[i]) else ""
            for i in np.argsort(-z)[:top_k]
        ]
        columns[label] = terms + [""] * (top_k - len(terms))
    return pd.DataFrame(columns, index=pd.RangeIndex(1, top_k + 1, name="rank"))


def _odds_ratio_and_z(present: spmatrix, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each term's odds ratio and the z-score of its log.

    Args:
        present: Binary document x term matrix.
        mask: True for the documents that carry the label.
    """
    in_label = np.asarray(present[mask].sum(axis=0)).ravel()
    in_rest = np.asarray(present[~mask].sum(axis=0)).ravel()
    # One 2x2 table per term; +0.5 per cell (Haldane-Anscombe) keeps empty cells finite.
    a = in_label + 0.5  # label, term present
    b = mask.sum() - in_label + 0.5  # label, term absent
    c = in_rest + 0.5  # rest, term present
    d = (~mask).sum() - in_rest + 0.5  # rest, term absent
    ratio = (a * d) / (b * c)
    return ratio, np.log(ratio) / np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def _is_empty(value: object) -> bool:
    """Tell whether a value is an empty string or an empty list."""
    return isinstance(value, (str, list)) and len(value) == 0
