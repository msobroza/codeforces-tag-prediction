"""The label prior: the F1 floor every other method is compared with (docs/adr/0004).

For each focus tag that at least ``MIN_POSITIVES`` problems of its fit data have, it predicts the
tag for every problem, and no other: it reads no text. Predicting a tag for every problem scores
F1 = 2p / (1 + p) on it, p being the tag's share of positives, the best a constant prediction
can do; predicting nothing scores 0. A kind "fitted" method (CONTEXT.md, Method kind): its dev
predictions come from ``cross_predict``, and its test predictions from the method fitted on all
of dev.

This module is the method's prediction path: it imports numpy and pandas, never mlflow or optuna.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.label_prior.settings import MIN_POSITIVES, NAME, PRIOR_FILE


@dataclass(frozen=True)
class LabelPrior:
    """A ``TaggingMethod`` that predicts the same focus tags for every problem.

    ``LabelPrior()`` is unfitted: it predicts nothing. ``fit`` returns a fitted copy.

    Attributes:
        predicted: The focus tags predicted for every problem, in ``FOCUS_TAGS`` order.
    """

    predicted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep each predicted tag once, in ``FOCUS_TAGS`` order.

        Raises:
            ValueError: If a predicted tag is not a focus tag.
        """
        unknown = sorted(set(self.predicted) - set(FOCUS_TAGS))
        if unknown:
            raise ValueError(
                f"{unknown} are not focus tags: label_prior predicts {list(FOCUS_TAGS)} only."
            )
        object.__setattr__(
            self, "predicted", tuple(tag for tag in FOCUS_TAGS if tag in self.predicted)
        )

    @property
    def name(self) -> str:
        """Its key in ``METHODS``."""
        return NAME

    @property
    def tags(self) -> tuple[str, ...]:
        """The output columns: the focus tags, in ``FOCUS_TAGS`` order."""
        return FOCUS_TAGS

    def fit(  # pylint: disable=unused-argument
        self, raw_problems: pd.DataFrame, labels: pd.DataFrame
    ) -> LabelPrior:
        """Fit on the labels: the prior reads no text, so ``raw_problems`` is not used.

        Args:
            raw_problems: The fit problems' raw rows, indexed by ``sample_id``.
            labels: 0/1 labels with the index of ``raw_problems`` and a column per focus tag.

        Returns:
            A method that predicts each focus tag with ``MIN_POSITIVES`` or more positives in
            ``labels``, for every problem.
        """
        positives = labels[list(FOCUS_TAGS)].sum()
        return LabelPrior(tuple(tag for tag in FOCUS_TAGS if positives[tag] >= MIN_POSITIVES))

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Give every problem the same tags.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``; no column is read.

        Returns:
            Int 0/1, with the index of ``raw_problems``, one column per focus tag.
        """
        row = np.array([int(tag in self.predicted) for tag in FOCUS_TAGS])
        return pd.DataFrame(
            np.tile(row, (len(raw_problems), 1)),
            index=raw_problems.index,
            columns=list(FOCUS_TAGS),
        )

    def save(self, folder: Path) -> None:
        """Write the predicted tags to ``label_prior.json`` in ``folder``.

        Args:
            folder: An existing directory.
        """
        text = json.dumps({"predicted": list(self.predicted)}, indent=2)
        (Path(folder) / PRIOR_FILE).write_text(text + "\n", encoding="utf-8")


def load(folder: Path) -> LabelPrior:
    """Load the label prior saved in a method folder: the ``load`` its registry entry names.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        The method.

    Raises:
        ValueError: If the file names a tag that is not a focus tag.
    """
    stored = json.loads((Path(folder) / PRIOR_FILE).read_text(encoding="utf-8"))
    return LabelPrior(tuple(stored["predicted"]))
