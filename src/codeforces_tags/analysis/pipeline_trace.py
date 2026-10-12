"""How the tag model turns a problem into tag scores, for notebook 04.

``parameter_table`` pairs each field of ``ModelParams`` with the parameter it sets on the tag
model's estimator (``TagModel.estimator``, a ``CrossFittedTagThresholds``). ``trace_problem``
follows one raw problem through the fitted pipeline, by slicing it: the texts and tokens each
block reads, the problem's non-zero features, what each adds to each tag's score, and each tag's
decision. It explains the mechanics of a fitted model, not its accuracy. Nothing here is on the
prediction path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from codeforces_tags.config import RAW_INPUT_COLUMNS
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.model.features import SAMPLE_IO_CUES, block_feature_names
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.tag_model import TagModel

NOT_USED = "not used: block not kept"
_CLASSIFIER = "estimator__classifier__estimator"
_FEATURES = "estimator__features"
# Each field of ModelParams: the block whose parameter it is (None: every configuration uses it)
# and the name that sets it on a CrossFittedTagThresholds (set_params).
_SETS: dict[str, tuple[str | None, str]] = {
    "classifier": (None, _CLASSIFIER),
    "c": (None, f"{_CLASSIFIER}__C"),
    "class_weight": (None, f"{_CLASSIFIER}__class_weight"),
    "include_notes": ("statement", f"{_FEATURES}__statement__text__kw_args"),
    "statement_ngram_max": ("statement", f"{_FEATURES}__statement__tfidf__ngram_range"),
    "statement_min_df": ("statement", f"{_FEATURES}__statement__tfidf__min_df"),
    "statement_max_df": ("statement", f"{_FEATURES}__statement__tfidf__max_df"),
    "statement_sublinear_tf": ("statement", f"{_FEATURES}__statement__tfidf__sublinear_tf"),
    "code_weight": ("code", f"{_FEATURES}__transformer_weights"),
    "code_ngram_max": ("code", f"{_FEATURES}__code__ngram_range"),
    "code_min_df": ("code", f"{_FEATURES}__code__min_df"),
    "code_sublinear_tf": ("code", f"{_FEATURES}__code__sublinear_tf"),
    "cue_weight": ("cues", f"{_FEATURES}__transformer_weights"),
    "blocks": (None, f"{_FEATURES}__transformers"),
}


def parameter_table(params: ModelParams) -> pd.DataFrame:
    """Pair each field of a configuration with the estimator parameter that sets it.

    The names are those of ``set_params`` on ``TagModel.estimator``, the fitted
    ``CrossFittedTagThresholds`` that ``load_method`` returns inside the tag model: for example,
    ``code_weight`` is set through ``estimator__features__transformer_weights``, the dict of
    block weights. A field of a block that the configuration does not keep is set nowhere.

    Args:
        params: A configuration, e.g. the frozen ``params.json``.

    Returns:
        One row per field of ``ModelParams``, in field order, indexed by ``parameter``: the
        field's ``value`` and ``sets``, the name of the parameter it sets, or ``NOT_USED``.
    """
    rows = {}
    for field in fields(ModelParams):
        block, name = _SETS[field.name]
        kept = block is None or block in params.blocks
        rows[field.name] = {
            "value": getattr(params, field.name),
            "sets": name if kept else NOT_USED,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("parameter")


@dataclass(frozen=True)
class PipelineTrace:
    """One problem followed through the fitted pipeline, from its raw fields to its tag scores.

    Attributes:
        steps: What the problem becomes at each step, in order: ``raw fields`` (a Series of the
            fields the model reads), then, for each block the pipeline keeps, its own steps:
            ``statement`` (the text the statement block reads: the statement, followed by the
            notes when the configuration includes them) and ``normalised statement`` (that text
            after the normaliser, as a string), ``code tokens`` (a list of tokens, before
            n-grams), ``sample I/O cues`` (a Series of the five cues, 1 when one fires).
        block_values: For each kept block, the problem's non-zero features and their values as
            the block computes them, before the block's weight. Indexed by ``feature``.
        weights: What each feature adds to each tag's score per unit of the feature, with its
            block's weight: ``TagModel.feature_weights()``.
        decision: Per tag (the index): the ``score``, the ``intercept`` and the ``threshold``
            of the tag's model, whether the tag is ``predicted`` (the score reaches the
            threshold), and whether it is ``true`` of the problem (0 or 1).
    """

    steps: dict[str, object]
    block_values: dict[str, pd.Series]
    weights: pd.DataFrame
    decision: pd.DataFrame

    def features(self, block: str) -> pd.Series:
        """Return the problem's non-zero features in a block, the largest value first.

        Args:
            block: One of the blocks the pipeline keeps.

        Returns:
            The features' values before the block's weight, indexed by ``feature``.

        Raises:
            ValueError: If the pipeline does not keep ``block``.
        """
        if block not in self.block_values:
            raise ValueError(
                f"The pipeline keeps no {block!r} block: it keeps {list(self.block_values)}."
            )
        return self.block_values[block].sort_values(ascending=False, kind="stable")

    def contributions(self, tag: str) -> pd.DataFrame:
        """Return what each of the problem's features adds to a tag's score, the largest first.

        A feature's contribution is its value times its weight, so the contributions plus the
        tag's intercept add up to its score (up to rounding).

        Args:
            tag: One of the tags the model predicts.

        Returns:
            One row per non-zero feature of every block, indexed by (``block``, ``feature``):
            its ``value`` before the block's weight, its ``weight`` for the tag and the
            ``contribution``, ranked by contribution.

        Raises:
            ValueError: If the model does not predict ``tag``.
        """
        if tag not in self.decision.index:
            raise ValueError(
                f"The model predicts no tag {tag!r}: pick one of {self.decision.index.tolist()}."
            )
        values = pd.concat(self.block_values, names=["block", "feature"])
        table = pd.DataFrame({"value": values, "weight": self.weights.loc[values.index, tag]})
        table["contribution"] = table["value"] * table["weight"]
        return table.sort_values("contribution", ascending=False, kind="stable")


def trace_problem(tag_model: TagModel, raw_problems: pd.DataFrame) -> PipelineTrace:
    """Follow one raw problem through the fitted pipeline of a tag model, step by step.

    The pipeline is sliced, not refitted: ``pipeline[:1]`` preprocesses the problem, and each
    block's fitted transformer then reads its columns. The problem is traced as the model sees
    it, so a dev problem that the model was fitted on shows its mechanics, not its accuracy.

    Args:
        tag_model: A fitted tag model, e.g. ``load_method(model_dir("tfidf"))``.
        raw_problems: The raw problem, one row with its ``tags`` column, as ``select_part`` gives
            it: ``dev_raw.iloc[[0]]``.

    Returns:
        The problem's steps, features, weights and decisions.

    Raises:
        ValueError: If there is not exactly one problem or it has no ``tags`` column, or as
            ``raw_to_problems`` does if the problem is already preprocessed.
    """
    _check_one_problem(raw_problems)
    pipeline = tag_model.pipeline
    problems = pipeline[:1].transform(raw_problems)
    features = pipeline.named_steps["features"]
    fitted = {name: (transformer, columns) for name, transformer, columns in features.transformers_}
    steps: dict[str, object] = {"raw fields": raw_problems.iloc[0][list(RAW_INPUT_COLUMNS)]}
    block_values = {}
    for block in tag_model.params.blocks:
        transformer, columns = fitted[block]
        rows = problems[columns]
        steps |= _BLOCK_STEPS[block](transformer, rows)
        names = block_feature_names(features, block)
        block_values[block] = _non_zero(transformer.transform(rows), names)
    return PipelineTrace(
        steps, block_values, tag_model.feature_weights(), _decision(tag_model, raw_problems)
    )


def _check_one_problem(raw_problems: pd.DataFrame) -> None:
    """Raise a ValueError unless the frame is one problem with its tags."""
    if len(raw_problems) != 1:
        raise ValueError(
            f"trace_problem traces one problem, not {len(raw_problems)}: pass one row, e.g. "
            "dev_raw.iloc[[0]]."
        )
    if "tags" not in raw_problems:
        raise ValueError(
            "trace_problem shows the problem's true tags: pass it with its tags column, as "
            "select_part gives it."
        )


def _statement_steps(block: Pipeline, rows: pd.DataFrame) -> dict[str, object]:
    """Return the text the statement block reads, and that text once normalised."""
    text = block["text"].transform(rows)[0]
    return {"statement": text, "normalised statement": block["tfidf"].build_preprocessor()(text)}


def _code_steps(vectorizer: TfidfVectorizer, source: pd.Series) -> dict[str, object]:
    """Return the tokens the code block reads in the solution, before n-grams."""
    document = vectorizer.build_preprocessor()(source.iloc[0])
    return {"code tokens": vectorizer.build_tokenizer()(document)}


def _cue_steps(cues: FunctionTransformer, rows: pd.DataFrame) -> dict[str, object]:
    """Return the five sample I/O cues of the problem: 1 when a cue fires."""
    values = cues.transform(rows)[0].astype(int)
    return {"sample I/O cues": pd.Series(values, index=list(SAMPLE_IO_CUES), name="value")}


# The steps each block adds to a trace, from its fitted transformer and the columns it reads.
_BLOCK_STEPS: dict[str, Callable[..., dict[str, object]]] = {
    "statement": _statement_steps,
    "code": _code_steps,
    "cues": _cue_steps,
}


def _non_zero(values: sparse.spmatrix | np.ndarray, names: list[str]) -> pd.Series:
    """Return the non-zero entries of a one-row feature matrix, by feature name."""
    row = values.toarray()[0] if sparse.issparse(values) else np.asarray(values)[0]
    present = np.flatnonzero(row)
    return pd.Series(row[present], index=pd.Index(names)[present].rename("feature"), name="value")


def _decision(tag_model: TagModel, raw_problems: pd.DataFrame) -> pd.DataFrame:
    """Return per tag the score, intercept, threshold, prediction and truth of the problem."""
    classifier = tag_model.pipeline.named_steps["classifier"]
    intercepts = [float(model.intercept_[0]) for model in classifier.estimators_]
    return pd.DataFrame(
        {
            "score": tag_model.scores(raw_problems).iloc[0],
            "intercept": pd.Series(intercepts, index=tag_model.thresholds.index),
            "threshold": tag_model.thresholds,
            "predicted": tag_model.predict(raw_problems).iloc[0],
            "true": multi_hot(raw_problems["tags"], tag_model.tags).iloc[0],
        }
    )
