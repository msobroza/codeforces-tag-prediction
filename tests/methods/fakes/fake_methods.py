"""A fake tagging method for the registry tests.

The tests import it as the top-level module ``fake_methods`` (``monkeypatch.syspath_prepend``
on this folder), never as ``tests.…``: ``import tests`` finds a dependency's installed
``tests`` package.
"""

import json
from pathlib import Path

import pandas as pd

CONSTANT_FILE = "constant.json"


class ConstantMethod:
    """Predicts the same tags for every problem; saves its settings in ``constant.json``."""

    def __init__(self, name, tags, predicted):
        """Predict ``predicted`` among ``tags`` for every problem, under the name ``name``."""
        self._name = name
        self._tags = tuple(tags)
        self._predicted = tuple(predicted)

    @property
    def name(self):
        """The method's registry key."""
        return self._name

    @property
    def tags(self):
        """The output columns, in order."""
        return self._tags

    def predict(self, raw_problems):
        """Return the constant 0/1 row for every problem, indexed like ``raw_problems``."""
        row = [int(tag in self._predicted) for tag in self._tags]
        return pd.DataFrame([row] * len(raw_problems), index=raw_problems.index, columns=self._tags)

    def save(self, folder):
        """Write ``constant.json`` into ``folder``."""
        values = {"name": self._name, "tags": self._tags, "predicted": self._predicted}
        (Path(folder) / CONSTANT_FILE).write_text(json.dumps(values), encoding="utf-8")


def load(folder):
    """Load the method saved in ``folder``."""
    values = json.loads((Path(folder) / CONSTANT_FILE).read_text(encoding="utf-8"))
    return ConstantMethod(values["name"], values["tags"], values["predicted"])


class ScoredMethod(ConstantMethod):  # pylint: disable=too-few-public-methods
    """A constant method that also scores: a ``ScoreMethod``."""

    def scores(self, raw_problems):
        """Return 1.0 for each predicted tag, 0.0 otherwise."""
        return self.predict(raw_problems).astype(float)

    def predict_from_scores(self, scores):
        """Predict the tags scored 1.0."""
        return (scores >= 1.0).astype(int)

    def tag_thresholds(self):
        """Stand for the threshold of every tag, 1.0: no registry test asks for it."""
