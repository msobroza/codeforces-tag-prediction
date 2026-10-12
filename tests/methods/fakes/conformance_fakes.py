"""Tagging methods of each kind for the conformance tests, and methods that break one rule each.

The tests import it as the top-level module ``conformance_fakes`` (``monkeypatch.syspath_prepend``
on this folder), and pass this folder as ``sys_path`` to the isolation subprocess, which imports
it under the same name. Every method saves ``fake.json``, its class and state, and ``load``
rebuilds it from there; the score method also saves its fitted ``CrossFittedTagThresholds`` with
skops.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.base import BaseEstimator

from codeforces_tags.cache import ArrayCache, cache_key
from codeforces_tags.config import ALL_TAGS, FOCUS_TAGS, SKOPS_COMPRESSION
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.pretrained import pretrained_root

FAKE_FILE = "fake.json"
MODEL_FILE = "model.skops"
# The types of a saved ScoreFake that skops does not trust by default, by import path.
TRUSTED_TYPES = [
    "codeforces_tags.model.thresholds.CrossFittedTagThresholds",
    "conformance_fakes.LengthScorer",
]
# The statement length of LengthCutFake's tags with no positive: never reached.
NEVER = 10**9
# The folder of the array cache that CacheReadingFake reads: a test sets it.
CACHE_DIR = None


def load(folder):
    """Load the fake saved in ``folder``: the ``load`` of every fake's ``MethodSpec.module``."""
    stored = json.loads((Path(folder) / FAKE_FILE).read_text(encoding="utf-8"))
    return globals()[stored["class"]].restore(Path(folder), stored["state"])


class Fake:
    """Saves its class and ``state()`` in ``fake.json``; ``load`` calls its ``restore``."""

    name = "fake"
    tags = FOCUS_TAGS

    def state(self):
        """Return the keyword arguments that rebuild the method, as JSON values."""
        return {}

    @classmethod
    def restore(cls, folder, state):  # pylint: disable=unused-argument
        """Rebuild the method saved in ``folder`` from its state."""
        return cls(**state)

    def save(self, folder):
        """Write ``fake.json`` into ``folder``."""
        stored = {"class": type(self).__name__, "state": self.state()}
        (Path(folder) / FAKE_FILE).write_text(json.dumps(stored), encoding="utf-8")


def _lengths(raw_problems):
    """Return a constant, the statement length and the solution's line count of each problem."""
    problems = raw_to_problems(raw_problems)
    return np.column_stack(
        [
            np.ones(len(problems)),
            problems["statement"].str.len().to_numpy(float) / 100,
            problems["source_code"].str.count("\n").to_numpy(float),
        ]
    )


class LengthScorer(BaseEstimator):
    """A numpy-only scorer under the score-estimator contract: least squares on two lengths."""

    def fit(self, X, y):
        """Fit one linear score per tag on raw rows and an int label matrix; set ``classes_``."""
        self.coef_ = np.linalg.lstsq(_lengths(X), y, rcond=None)[0]
        self.classes_ = np.arange(y.shape[1])
        return self

    def decision_function(self, X):
        """Return one score per problem and tag, 0 at the halfway label."""
        return _lengths(X) @ self.coef_ - 0.5

    def predict(self, X):
        """Predict the tags whose score is at least 0."""
        return (self.decision_function(X) >= 0).astype(int)


class ScoreFake(Fake):
    """Kind "score": a fitted ``CrossFittedTagThresholds`` around a ``LengthScorer``."""

    name = "score_fake"

    def __init__(self, model):
        """Wrap a fitted ``CrossFittedTagThresholds``."""
        self.model = model

    def predict(self, raw_problems):
        """Predict the tags whose score reaches the model's threshold."""
        return self.predict_from_scores(self.scores(raw_problems))

    def scores(self, raw_problems):
        """Return the model's scores."""
        scores = self.model.decision_function(raw_problems)
        return pd.DataFrame(scores, index=raw_problems.index, columns=list(self.tags))

    def predict_from_scores(self, scores):
        """Predict the tags whose score reaches the model's threshold."""
        return (scores >= self.model.thresholds_).astype(int)

    def tag_thresholds(self):
        """Return the model's thresholds, by tag."""
        return pd.Series(self.model.thresholds_, index=list(self.tags), dtype=float)

    def save(self, folder):
        """Save the model with skops, then ``fake.json``."""
        sio.dump(self.model, Path(folder) / MODEL_FILE, compression=SKOPS_COMPRESSION)
        super().save(folder)

    @classmethod
    def restore(cls, folder, state):
        """Load the model, trusting only its two types."""
        return cls(sio.load(folder / MODEL_FILE, trusted=TRUSTED_TYPES), **state)


class LengthCutFake(Fake):
    """Kind "fitted": predicts a focus tag for statements as long as its shortest positive's."""

    name = "length_cut_fake"

    def __init__(self, cuts=None):
        """Predict each tag for statements of at least ``cuts[tag]`` characters."""
        self.cuts = cuts

    def fit(self, raw_problems, labels):
        """Return the method fitted on raw problems and their 0/1 focus labels."""
        lengths = raw_to_problems(raw_problems)["statement"].str.len()
        cuts = {
            tag: int(lengths[labels[tag] == 1].min()) if labels[tag].any() else NEVER
            for tag in self.tags
        }
        return type(self)(cuts)

    def predict(self, raw_problems):
        """Predict each tag whose cut the statement reaches."""
        lengths = raw_to_problems(raw_problems)["statement"].str.len()
        predicted = {tag: lengths.ge(self.cuts[tag]).astype(int) for tag in self.tags}
        return pd.DataFrame(predicted, index=raw_problems.index)

    def state(self):
        """Return the cuts."""
        return {"cuts": self.cuts}


class KeywordFake(Fake):
    """Kind "zero_shot", no API call: predicts the tags named in the statement or the notes."""

    name = "keyword_fake"
    tags = ALL_TAGS

    def predict(self, raw_problems):
        """Predict each tag whose name the statement or the notes hold."""
        problems = raw_to_problems(raw_problems)
        text = (problems["statement"] + " " + problems["prob_desc_notes"].fillna("")).str.lower()
        predicted = {tag: text.str.contains(tag, regex=False).astype(int) for tag in self.tags}
        return pd.DataFrame(predicted, index=raw_problems.index)


class FirstRowFake(KeywordFake):
    """A KeywordFake that, in the isolation subprocess, accepts only problem 1, without notes.

    The subprocess runs ``python -c``: there, any other problem means the check sent another
    row than the first raw problem it was given, and the assertion fails the subprocess.
    """

    def predict(self, raw_problems):
        """Predict as KeywordFake, after checking the subprocess's problem."""
        first = list(raw_problems.index) == [1] and raw_problems["prob_desc_notes"].isna().all()
        assert sys.argv[0] != "-c" or first, f"the subprocess got {list(raw_problems.index)}"
        return super().predict(raw_problems)


class CodeRuleFake(Fake):
    """Kind "other": rules on the raw description and solution, the only columns it reads."""

    name = "code_rule_fake"
    tags = tuple(tag for tag in ALL_TAGS if tag in {*FOCUS_TAGS, "implementation"})

    def predict(self, raw_problems):
        """Predict "trees" from the description and "implementation" from long solutions."""
        predicted = pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags))
        description = raw_problems["prob_desc_description"].str.lower()
        predicted["trees"] = description.str.contains("tree", regex=False).astype(int)
        predicted["implementation"] = (raw_problems["source_code"].str.count("\n") >= 2).astype(int)
        return predicted


class ZerosFake(Fake):
    """Predicts no tag. Its tags are given, so a test can break one tag rule at a time.

    It saves no state: a loaded one has the focus tags.
    """

    name = "zeros_fake"

    def __init__(self, tags=FOCUS_TAGS):
        """Predict zeros for ``tags``."""
        self.tags = tags

    def predict(self, raw_problems):
        """Return 0 for every problem and tag."""
        return pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags))


class MalformedFake(ZerosFake):
    """Returns ``change(predictions)``: a test passes the change that breaks an output rule."""

    def __init__(self, change):
        """Apply ``change`` to every prediction."""
        super().__init__()
        self.change = change

    def predict(self, raw_problems):
        """Return the changed zeros."""
        return self.change(super().predict(raw_problems))


class MutatingFake(ZerosFake):
    """Writes into the raw problems it is given."""

    def predict(self, raw_problems):
        """Blank the solutions, then predict zeros."""
        raw_problems["source_code"] = ""
        return super().predict(raw_problems)


class BatchFake(ZerosFake):
    """Predicts "math" for every problem of a batch of two or more: a row depends on the others."""

    def predict(self, raw_problems):
        """Predict "math" unless given one problem."""
        return super().predict(raw_problems).assign(math=int(len(raw_problems) > 1))


class LeakyFake(ZerosFake):
    """Predicts "math" from the true tags whenever the raw problems carry them."""

    def predict(self, raw_problems):
        """Copy the "math" label when the ``tags`` column is there."""
        predicted = super().predict(raw_problems)
        if "tags" in raw_problems:
            predicted["math"] = raw_problems["tags"].map(lambda tags: int("math" in tags))
        return predicted


class ForgetfulFake(LengthCutFake):
    """Saves cuts that are never reached: once loaded, it predicts no tag."""

    def state(self):
        """Return unreachable cuts."""
        return {"cuts": dict.fromkeys(self.tags, NEVER)}


class EdgeFake(Fake):
    """Kind "score": scores 0, or ``offset`` when given one problem; predicts at ``threshold``.

    Its threshold is ``moved`` higher for a problem scored alone: each call's tags are its scores
    at that call's ``tag_thresholds()``, yet with ``moved`` set a tag flips between a batch and
    one problem against its scores.
    """

    name = "edge_fake"

    def __init__(self, offset, threshold, moved=0.0):
        """Score one problem ``offset`` above a batch, at ``threshold`` (``moved`` higher alone)."""
        self.offset = offset
        self.threshold = threshold
        self.moved = moved
        self.alone = False

    def scores(self, raw_problems):
        """Return ``offset`` for a single problem, 0.0 within a batch, and remember which."""
        self.alone = len(raw_problems) == 1
        value = self.offset if self.alone else 0.0
        return pd.DataFrame(value, index=raw_problems.index, columns=list(self.tags))

    def predict(self, raw_problems):
        """Predict the tags whose score reaches the threshold."""
        return self.predict_from_scores(self.scores(raw_problems))

    def predict_from_scores(self, scores):
        """Predict the tags whose score reaches the threshold of the problems scored last."""
        return (scores >= self._threshold()).astype(int)

    def tag_thresholds(self):
        """Return the threshold of the problems scored last, for every tag."""
        return pd.Series(self._threshold(), index=list(self.tags), dtype=float)

    def state(self):
        """Return the offset, the threshold and its move."""
        return {"offset": self.offset, "threshold": self.threshold, "moved": self.moved}

    def _threshold(self):
        """Return ``threshold``, ``moved`` higher when the problems scored last were one."""
        return self.threshold + self.moved if self.alone else self.threshold


class LogProbabilityFake(EdgeFake):
    """An EdgeFake whose first tag scores -inf, as the log-probability of an impossible tag."""

    def scores(self, raw_problems):
        """Return EdgeFake's scores, -inf for the first tag."""
        return super().scores(raw_problems).assign(**{self.tags[0]: -np.inf})


class DisagreeingFake(EdgeFake):
    """Predicts as EdgeFake, but ``predict_from_scores`` gives the other tags of each problem."""

    def predict(self, raw_problems):
        """Predict from the scores as EdgeFake does."""
        return EdgeFake.predict_from_scores(self, self.scores(raw_problems))

    def predict_from_scores(self, scores):
        """Return the opposite of EdgeFake's predictions."""
        return 1 - super().predict_from_scores(scores)


class ScoreMutatingFake(EdgeFake):
    """Predicts from its scores as EdgeFake does, then overwrites the scores it was given."""

    def predict_from_scores(self, scores):
        """Return EdgeFake's predictions, after setting every given score to -1.0."""
        predictions = super().predict_from_scores(scores)
        scores.loc[:, :] = -1.0
        return predictions


class RaisingFromScoresFake(EdgeFake):
    """Predicts as EdgeFake, but ``predict_from_scores`` fails on the scores it is given."""

    def predict(self, raw_problems):
        """Predict from the scores as EdgeFake does."""
        return EdgeFake.predict_from_scores(self, self.scores(raw_problems))

    def predict_from_scores(self, scores):
        """Fail, as a method expecting other columns would."""
        raise KeyError("expected other columns")


class ScoresOnlyFake(ZerosFake):
    """Registered as a score method and scores, but cannot predict from given scores."""

    name = "edge_fake"

    def scores(self, raw_problems):
        """Score every tag 0.0."""
        return self.predict(raw_problems).astype(float)


class MalformedScoresFake(EdgeFake):
    """Returns ``change(scores)`` from scores, while predict reads the right scores."""

    def __init__(self, change):
        """Apply ``change`` to every score frame."""
        super().__init__(offset=0.0, threshold=0.5)
        self.change = change

    def scores(self, raw_problems):
        """Return the changed scores."""
        return self.change(super().scores(raw_problems))

    def predict(self, raw_problems):
        """Predict the tags whose unchanged score reaches the threshold."""
        return (EdgeFake.scores(self, raw_problems) >= self.threshold).astype(int)


class MalformedThresholdsFake(EdgeFake):
    """Predicts as EdgeFake, but ``tag_thresholds`` returns ``change(thresholds)``."""

    def __init__(self, change):
        """Apply ``change`` to the thresholds it returns."""
        super().__init__(offset=0.0, threshold=0.5)
        self.change = change

    def tag_thresholds(self):
        """Return the changed thresholds."""
        return self.change(super().tag_thresholds())


class ScorelessLoadFake(EdgeFake):
    """Loads as a method of the same name and tags, but without scores."""

    @classmethod
    def restore(cls, folder, state):
        """Return zeros under this method's name."""
        method = ZerosFake()
        method.name = cls.name
        return method


class ImportingFake(ZerosFake):
    """Imports ``modules`` when loaded, as a method whose prediction path imports them."""

    name = "importing_fake"

    def __init__(self, modules):
        """Import ``modules`` at load time."""
        super().__init__()
        self.modules = modules

    def state(self):
        """Return the modules to import."""
        return {"modules": list(self.modules)}

    @classmethod
    def restore(cls, folder, state):
        """Import the modules, then rebuild the method."""
        for module in state["modules"]:
            importlib.import_module(module)
        return super().restore(folder, state)


class PretrainedFake(ZerosFake):
    """Reads ``weights.json`` from the pretrained-weight cache when loaded."""

    name = "pretrained_fake"

    @classmethod
    def restore(cls, folder, state):
        """Read the weights from the pretrained cache, ``pretrained_root()``, then rebuild."""
        json.loads((pretrained_root() / "weights.json").read_text(encoding="utf-8"))
        return super().restore(folder, state)


class PickleFileFake(ZerosFake):
    """Also writes a file with a pickle file name into its method folder."""

    name = "pickle_file_fake"

    def save(self, folder):
        """Write ``fake.json``, then ``state.pkl``."""
        super().save(folder)
        (Path(folder) / "state.pkl").write_text("{}", encoding="utf-8")


class ApiFake(ZerosFake):
    """Stands for a method that sends paid requests: its predict must not run in the check."""

    name = "api_fake"

    def predict(self, raw_problems):
        """Refuse to run, as a paid request would."""
        raise RuntimeError("api_fake would send a paid request.")


class CacheReadingFake(ZerosFake):
    """Reads the array cache when it predicts: its timed calls would run on cached arrays."""

    name = "cache_reading_fake"

    def predict(self, raw_problems):
        """Fetch one array twice, the second time from the cache, then predict zeros."""
        cache = ArrayCache(CACHE_DIR)
        key = cache_key(self.name)
        cache.get_or_compute(key, lambda: np.zeros(1))
        cache.get_or_compute(key, lambda: np.zeros(1))
        return super().predict(raw_problems)
