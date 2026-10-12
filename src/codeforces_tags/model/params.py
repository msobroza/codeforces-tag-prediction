"""One configuration of the tag model: the feature blocks it keeps, their settings, its classifier.

``ModelParams`` is what a tuning trial samples, what ``params.json`` stores and what the
pipeline is built from. The classifier, class weight, blocks, C and block weights are checked
when it is created; scikit-learn checks the TF-IDF settings at fit.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, fields

BLOCKS = ("statement", "code", "cues")
CLASSIFIERS = ("logistic_regression", "linear_svm")
CLASS_WEIGHTS = ("none", "balanced")


@dataclass(frozen=True)
class ModelParams:  # pylint: disable=too-many-instance-attributes
    """One configuration of the tag model, a point of the search space.

    Attributes:
        classifier: The linear model fitted for each tag, one of ``CLASSIFIERS``.
        c: Inverse regularisation strength of that model (sklearn's ``C``), above 0.
        class_weight: ``"balanced"`` weights each problem, in each tag's binary model, by
            n / (2 n_class) for its class; ``"none"`` weights every problem 1.
        include_notes: Append the notes to the statement block's text.
        statement_ngram_max: Longest word n-gram of the statement block.
        statement_min_df: Words of fewer problems than this are dropped from the statement block.
        statement_max_df: Words of a larger share of problems than this are dropped.
        statement_sublinear_tf: Count a word 1 + log(tf) times instead of tf times.
        code_weight: Weight of the code block (the statement block's is 1), above 0 when kept.
        code_ngram_max: Longest token n-gram of the code block.
        code_min_df: Tokens of fewer problems than this are dropped from the code block.
        code_sublinear_tf: Count a token 1 + log(tf) times instead of tf times.
        cue_weight: Weight of the cue block, above 0 when kept.
        blocks: The feature blocks kept, stored as a tuple in ``BLOCKS`` order.
    """

    classifier: str
    c: float
    class_weight: str
    include_notes: bool
    statement_ngram_max: int
    statement_min_df: int
    statement_max_df: float
    statement_sublinear_tf: bool
    code_weight: float
    code_ngram_max: int
    code_min_df: int
    code_sublinear_tf: bool
    cue_weight: float
    blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check the values, and store the blocks as a tuple in ``BLOCKS`` order.

        Raises:
            ValueError: If the classifier or class weight is unknown; if the blocks are empty,
                repeat a block or name one outside ``BLOCKS``; if ``c`` is not above 0; or if
                the weight of a kept code or cue block is not above 0.
        """
        _check_choice("classifier", self.classifier, CLASSIFIERS)
        _check_choice("class_weight", self.class_weight, CLASS_WEIGHTS)
        _check_blocks(self.blocks)
        object.__setattr__(self, "blocks", tuple(block for block in BLOCKS if block in self.blocks))
        _check_positive("c", self.c)
        if "code" in self.blocks:
            _check_positive("code_weight", self.code_weight)
        if "cues" in self.blocks:
            _check_positive("cue_weight", self.cue_weight)

    def to_dict(self) -> dict[str, object]:
        """Return the fields as a JSON-able dict, the blocks as a list.

        Returns:
            One entry per field, in field order.
        """
        return asdict(self) | {"blocks": list(self.blocks)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ModelParams:
        """Build the parameters from a dict such as ``to_dict`` returns, read back from JSON.

        Args:
            values: One entry per field, exactly; ``blocks`` may be a list, in any order.

        Returns:
            The checked parameters.

        Raises:
            ValueError: If a key is not a field or a field has no key (the message names
                both), or if a value is invalid.
        """
        names = [field.name for field in fields(cls)]
        unknown = sorted(set(values) - set(names))
        missing = sorted(set(names) - set(values))
        if unknown or missing:
            raise ValueError(f"ModelParams keys differ: unknown {unknown}, missing {missing}.")
        return cls(**values)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Raise a ValueError naming ``name`` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, not {value!r}.")


def _check_blocks(blocks: Collection[str]) -> None:
    """Raise a ValueError unless the blocks are distinct names from ``BLOCKS``, at least one."""
    if not blocks:
        raise ValueError(f"Keep at least one block of {BLOCKS}.")
    if len(set(blocks)) < len(blocks):
        raise ValueError(f"Blocks {tuple(blocks)} name a block more than once.")
    unknown = sorted(set(blocks) - set(BLOCKS))
    if unknown:
        raise ValueError(f"Unknown blocks {unknown}: expected some of {BLOCKS}.")


def _check_positive(name: str, value: float) -> None:
    """Raise a ValueError naming ``name`` unless ``value`` is above 0 (NaN is not)."""
    if not value > 0:
        raise ValueError(f"{name} must be above 0, not {value}.")
