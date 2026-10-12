"""``JevTagger``: the score method that asks TypeSafe's Jev about each focus tag (docs/adr/0007).

A Jev method sends one request per problem (``questions.requests``): the problem's state and
one Noul question per focus tag. Jev answers a probability per question, which is the method's
score. ``JevProbabilities``, the scorer of ``CrossFittedTagThresholds`` that learns nothing,
passes the probabilities through, so a tag is predicted when its probability reaches the tag's
cross-fitted threshold, chosen on the dev answers at freeze. A frozen method sends the seed
words; the searched method sends the words its search chose (docs/adr/0010). A calibrated twin
asks Jev its base's request, with its base's words, and applies the thresholds its threshold
rule chose on its base's answers (docs/adr/0011).

A prediction opens one client session (``client.jev_session``) in an event loop of its own,
sends paid requests and writes no answers file: the frozen dev and test predictions come from
the committed answers files. The method folder holds ``jev.json`` (the method, the model, the
hash of the settings and the thresholds) and, for the searched method and its twin,
``words.json`` (their words, which that hash covers). Loading it imports no SDK and reads no
key: the SDK is imported only when a session opens.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.methods.typesafe_jev import client, questions
from codeforces_tags.methods.typesafe_jev import words as jev_words
from codeforces_tags.methods.typesafe_jev.answers import probabilities
from codeforces_tags.methods.typesafe_jev.settings import (
    CONCURRENCY,
    GEPA_NAME,
    METHOD_FILE_NAME,
    MODEL,
    NAMES,
    WORDS_FILE_NAME,
)
from codeforces_tags.methods.typesafe_jev.words import (
    JevWords,
    base_of,
    method_settings,
    read_words,
)
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.provenance import json_safe, sha256_json


class JevProbabilities(BaseEstimator):
    """The scorer of ``CrossFittedTagThresholds`` that learns nothing: Jev's probabilities.

    Its rows already hold a probability per focus tag, read from Jev's answers: ``fit`` learns
    nothing and ``decision_function`` returns them. It follows the score-estimator contract of
    ``CrossFittedTagThresholds`` and takes no parameter, so ``clone`` copies it.
    """

    def fit(  # pylint: disable=unused-argument
        self, X: pd.DataFrame, y: np.ndarray | pd.DataFrame
    ) -> JevProbabilities:
        """Learn nothing, but set ``classes_`` as the score-estimator contract asks.

        Args:
            X: The problems' probabilities, one column per focus tag.
            y: Their 0/1 labels, one column per tag: an (n, n_tags) array or a frame.

        Returns:
            The scorer itself.
        """
        self.classes_ = np.arange(np.shape(y)[1])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probabilities of the focus tags.

        Args:
            X: The problems' probabilities, one column per focus tag in any order; other
                columns are left out.

        Returns:
            An (n, n_tags) float array, its columns in ``FOCUS_TAGS`` order.
        """
        return X[list(FOCUS_TAGS)].to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the tags whose score is at least 0, as the score-estimator contract asks.

        ``CrossFittedTagThresholds`` never calls it: it applies its own thresholds.

        Args:
            X: The problems' probabilities, as ``decision_function`` takes them.

        Returns:
            An (n, n_tags) array of 0/1 ints.
        """
        return (self.decision_function(X) >= 0).astype(int)


def fitted_thresholds(thresholds: Sequence[float], tags: Sequence[str]) -> CrossFittedTagThresholds:
    """Rebuild the wrapper fitted at freeze from its thresholds.

    Its scorer learns nothing, so the scorer refit at freeze is a ``JevProbabilities`` with one
    class per tag. The rebuilt wrapper holds what ``predict`` and ``decision_function`` read:
    ``estimator_``, ``thresholds_`` and ``tags_``; the out-of-fold attributes stay with the
    wrapper fitted at freeze.

    Args:
        thresholds: One threshold per tag, in the order of ``tags``; ``inf`` for a tag that no
            dev problem has.
        tags: The tags, in order: ``FOCUS_TAGS``.

    Returns:
        The fitted ``CrossFittedTagThresholds(JevProbabilities())``.

    Raises:
        ValueError: If there is not one threshold per tag.
    """
    if len(thresholds) != len(tags):
        raise ValueError(
            f"{len(thresholds)} thresholds for {len(tags)} tags: the wrapper fitted at freeze "
            "holds one threshold per tag."
        )
    scorer = JevProbabilities()
    scorer.classes_ = np.arange(len(tags))
    wrapper = CrossFittedTagThresholds(JevProbabilities())
    wrapper.estimator_ = scorer
    wrapper.thresholds_ = np.asarray(thresholds, dtype=float)
    wrapper.tags_ = tuple(tags)
    return wrapper


class JevTagger:
    """A score method that asks Jev, then applies its cross-fitted thresholds (docs/adr/0007).

    Its scores are Jev's probabilities, 0.0 on every tag of a failed answer, and a tag is
    predicted when its probability reaches its threshold (``>=``). Each ``predict`` and
    ``scores`` call opens one client session, in an event loop of its own, and writes no answers
    file.

    Attributes:
        name: Its key in ``METHODS``: one of ``NAMES``, which says what state Jev reads.
        tags: The focus tags, in ``FOCUS_TAGS`` order.
        thresholds: The fitted wrapper: ``CrossFittedTagThresholds(JevProbabilities())`` at
            freeze, ``PlattLiptonThresholds`` for a twin at freeze, or ``fitted_thresholds`` of
            the stored thresholds once loaded.
        words: The words its questions hold: a calibrated twin's are its base's.
    """

    tags: tuple[str, ...] = FOCUS_TAGS

    def __init__(
        self, name: str, thresholds: CrossFittedTagThresholds, words: JevWords | None = None
    ) -> None:
        """Bind the method's name, its fitted thresholds and its words.

        Args:
            name: One of ``NAMES``.
            thresholds: The wrapper fitted at freeze, or ``fitted_thresholds`` of its
                thresholds.
            words: The words it sends; None for a frozen method or its calibrated twin, which
                send the seed words.

        Raises:
            ValueError: If ``name`` is not the name of a Jev method, the searched method or its
                twin is given no words, or a frozen method or its twin other words than the seed
                words.
        """
        if name not in NAMES:
            raise ValueError(f"{name!r} is not a Jev method: expected one of {list(NAMES)}.")
        if words is None:
            if base_of(name) == GEPA_NAME:
                raise ValueError(
                    f"{name} needs its words: they are those the search of {GEPA_NAME} chose, "
                    f"which its method folder holds in {WORDS_FILE_NAME}."
                )
            words = jev_words.SEED_WORDS
        method_settings(name, words)  # refuses other words than the seed's for a frozen method
        self.name = name
        self.thresholds = thresholds
        self.words = words

    async def apredict_with_answers(
        self, raw_problems: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[LlmAnswer]]:
        """Predict the tags of raw problems in the running event loop, and keep Jev's answers.

        At most ``CONCURRENCY`` requests are in flight, through one session; no problem opens
        no session. ``tracking latency`` times it on one problem at a time.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            The predictions, int 0/1 with the index of ``raw_problems`` and one column per tag,
            and the answers, one per problem in its order (for their tokens, cost and latency).

        Raises:
            ValueError: As ``questions.requests``, e.g. for a problem without a solution.
            TransientApiError: If Jev did not answer a request. Every request ends first, then
                the error of the first problem whose request failed is raised, whatever it is:
                a prediction skips no problem.
        """
        answers = await self._answers(raw_problems)
        return self.predict_from_scores(
            self.scores_from_answers(answers, raw_problems.index)
        ), answers

    def scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Score raw problems with Jev's probabilities, in an event loop of its own.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Floats with the index of ``raw_problems`` and one column per tag: the
            ``decision_function`` of ``thresholds``, i.e. the probabilities.

        Raises:
            ValueError: As ``apredict_with_answers``.
            TransientApiError: As ``apredict_with_answers``.
        """
        answers = asyncio.run(self._answers(raw_problems))
        return self.scores_from_answers(answers, raw_problems.index)

    def scores_from_answers(self, answers: Sequence[LlmAnswer], index: pd.Index) -> pd.DataFrame:
        """Score problems from Jev's answers already received: no request is sent.

        The command line reads a prediction's scores from the answers that gave its tags
        (``apredict_with_answers``): a second request would cost again, and Jev would answer it
        with a new sample.

        Args:
            answers: One answer per problem of ``index``, in any order.
            index: The ``sample_id`` of each problem, in the order wanted.

        Returns:
            What ``scores`` returns for those problems: floats indexed by ``index``, one column
            per tag, 0.0 on every tag of a failed answer.

        Raises:
            ValueError: As ``answers.probabilities``, e.g. for a problem without an answer.
        """
        found = probabilities(answers, index)
        return self._frame(self.thresholds.decision_function(found), index)

    def predict_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of problems from their scores, Jev's probabilities: no request is sent.

        Args:
            scores: What ``scores`` or ``scores_from_answers`` returned.

        Returns:
            Int 0/1, with the index of ``scores`` and one column per tag: 1 where the
            probability reaches the tag's threshold.
        """
        return self._frame(self.thresholds.predict(scores), scores.index)

    def tag_thresholds(self) -> pd.Series:
        """Return the per-tag thresholds that ``predict_from_scores`` applies (``>=``).

        Returns:
            The fitted wrapper's ``thresholds_`` as floats indexed by the focus tags, in
            ``FOCUS_TAGS`` order: ``inf`` for a tag that no dev problem has.
        """
        return pd.Series(self.thresholds.thresholds_, index=list(self.tags), dtype=float)

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of raw problems, in an event loop of its own.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Int 0/1, with the index of ``raw_problems`` and one column per tag: 1 where the
            probability reaches the tag's threshold.

        Raises:
            ValueError: As ``apredict_with_answers``.
            TransientApiError: As ``apredict_with_answers``.
        """
        predictions, _ = asyncio.run(self.apredict_with_answers(raw_problems))
        return predictions

    def save(self, folder: Path) -> None:
        """Write ``jev.json`` into ``folder``, and the words of the searched method or its twin.

        ``jev.json`` holds the name, the model, the hash of ``method_settings`` (its words
        included) and the thresholds. JSON has no infinity: the threshold of a tag that no dev
        problem has is the string ``"inf"`` (``json_safe``). ``words.json`` holds the words of
        the searched method, or of its calibrated twin (``JevWords.settings``), indented,
        non-ASCII text kept.

        Args:
            folder: An existing directory.
        """
        stored = {
            "method": self.name,
            "model": MODEL,
            "settings_sha256": sha256_json(method_settings(self.name, self.words)),
            "tags": list(self.tags),
            "thresholds": json_safe([float(value) for value in self.thresholds.thresholds_]),
        }
        text = json.dumps(stored, indent=2, allow_nan=False)
        (Path(folder) / METHOD_FILE_NAME).write_text(text + "\n", encoding="utf-8")
        if base_of(self.name) == GEPA_NAME:
            words = json.dumps(self.words.settings(), indent=2, ensure_ascii=False)
            (Path(folder) / WORDS_FILE_NAME).write_text(words + "\n", encoding="utf-8")

    async def _answers(self, raw_problems: pd.DataFrame) -> list[LlmAnswer]:
        """Ask Jev the request of each problem through one session, or none for no problem."""
        requests = questions.requests(raw_problems, self.name, self.words)
        if not requests:
            return []
        async with client.jev_session(self.name) as ask:
            return await _answer_each(requests, ask)

    def _frame(self, values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """Return the values of each problem and tag as a frame with the problems' index."""
        return pd.DataFrame(values, index=index, columns=list(self.tags))


def load(folder: Path) -> JevTagger:
    """Load the Jev method saved in a method folder: the ``load`` its registry entries name.

    It imports no SDK and reads no key: a session opens only when the method predicts.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        The method, with the thresholds chosen at freeze, and its words: the seed words for a
        frozen method or its twin, those of the folder's ``words.json`` for the searched method
        or its twin.

    Raises:
        ValueError: If ``jev.json`` names no Jev method, or holds another model, another hash
            of the settings (the searched method's words included) or other tags than this code
            and the folder's words: its thresholds were chosen on the answers of another method;
            or as ``words.read_words``.
        FileNotFoundError: If the folder of the searched method or its twin holds no
            ``words.json``.
    """
    path = Path(folder) / METHOD_FILE_NAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    name = _stored_name(path, stored)
    if base_of(name) == GEPA_NAME:
        words = read_words(Path(folder) / WORDS_FILE_NAME)
    else:
        words = jev_words.SEED_WORDS
    _check_stored(path, stored, method_settings(name, words))
    thresholds = [float(value) for value in stored["thresholds"]]
    return JevTagger(name, fitted_thresholds(thresholds, FOCUS_TAGS), words)


def _stored_name(path: Path, stored: dict) -> str:
    """Return the method a ``jev.json`` names, unless it names no Jev method."""
    if stored["method"] not in NAMES:
        raise ValueError(
            f"{path} holds the method {stored['method']!r}, which is not a Jev method: expected "
            f"one of {list(NAMES)}."
        )
    return stored["method"]


def _check_stored(path: Path, stored: dict, settings: dict[str, object]) -> None:
    """Refuse a ``jev.json`` of another model, settings or tags than the method's."""
    code = {"model": MODEL, "settings_sha256": sha256_json(settings), "tags": list(FOCUS_TAGS)}
    found = {key: stored[key] for key, value in code.items() if stored[key] != value}
    if found:
        wanted = {key: code[key] for key in found}
        raise ValueError(
            f"{path} holds {found}, not this code's {wanted}: its thresholds were chosen on the "
            "answers of another method (docs/adr/0003)."
        )


async def _answer_each(
    requests: Sequence[questions.JevRequest], ask: client.Ask
) -> list[LlmAnswer]:
    """Send every request, ``CONCURRENCY`` at most at once, then raise the first error in order.

    Every request ends before an error is raised, so the session never closes under a request
    in flight.
    """
    slots = asyncio.Semaphore(CONCURRENCY)

    async def answer(request: questions.JevRequest) -> LlmAnswer:
        async with slots:
            return await ask(request)

    outcomes = await asyncio.gather(*map(answer, requests), return_exceptions=True)
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if errors:
        raise errors[0]
    return outcomes
