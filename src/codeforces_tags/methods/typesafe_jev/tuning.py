"""The search over Jev's words: GEPA's adapter, its stoppers and its recorder (docs/adr/0010).

``run_search`` runs ``gepa.optimize`` over the nine pieces of ``jev-statement-code``'s words
(``words.PIECES``), from the seed candidate (the frozen words), with the learning folds as
GEPA's trainset and the selection fold as its valset. The held-out fold is never read:
``search_data`` drops it, and the adapter and the Jev cache refuse its problems.

- ``JevSearchAdapter`` scores a candidate with Jev's probabilities (``jev_cache.JevCache``) and
  the balanced Brier objective (``objective``), builds each piece's reflective records and asks
  the reflection model for a rewrite (``reflection``). A refused rewrite leaves its piece out;
  a proposal with no piece makes GEPA skip the child, at no Jev cost.
- ``IterationCap`` and ``CostCap`` stop the search between iterations.
- ``SearchRecorder``, a GEPA callback, writes ``history.jsonl`` (one line per iteration) and
  ``candidates.jsonl`` (one line per candidate scored on the selection fold) from the start on
  every run, and ``outcome.json`` (how the search ended, and the caps this run used) once a
  cap has stopped it.

Every answer comes from a cache once it has been paid for, and GEPA's choices come from its own
seeded random generator and the seeded sampler: running the search again replays it at no cost,
then goes on. GEPA's state file is a pickle, so no ``run_dir`` is passed; GEPA's own reflection
model, logger and trackers are unused. GEPA catches every ``Exception`` raised while it builds a
reflective dataset or proposes texts, and in a callback, and goes on: the adapter carries such an
error past GEPA (``_Escape``), and the recorder keeps its own until the next evaluation raises it.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import gepa
import pandas as pd
from gepa.core.adapter import EvaluationBatch

from codeforces_tags.config import EXPERIMENTS_DIR, FOCUS_TAGS
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.llm.answers import answer_cost_usd, problem_list
from codeforces_tags.llm.client import Ask, LlmAnswer, TransientApiError
from codeforces_tags.methods.typesafe_jev.jev_cache import (
    SEARCH_FOLDS,
    JevCache,
    JevSession,
    SearchInterrupted,
)
from codeforces_tags.methods.typesafe_jev.objective import (
    balanced_brier,
    problem_scores,
    tag_weights,
    weighted_errors,
)
from codeforces_tags.methods.typesafe_jev.reflection import (
    Reflector,
    convention_records,
    read_proposal,
    reflection_request,
    tag_records,
    texts_shown,
)
from codeforces_tags.methods.typesafe_jev.sampling import StratifiedSampler
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_LINES_FILE,
    GEPA_NAME,
    HISTORY_LINES_FILE,
    JEV_CACHE_FOLDER,
    OUTCOME_FILE,
    REFLECTIONS_FILE,
    SEARCH_FOLDER,
    SEARCH_GEPA_OPTIONS,
    SEARCH_HELD_OUT_FOLD,
    SEARCH_LEARNING_FOLDS,
    SEARCH_MAX_ITERATIONS,
    SEARCH_MINIBATCH_SIZE,
    SEARCH_POSITIVES_PER_TAG,
    SEARCH_SEED,
    SEARCH_SELECTION_FOLD,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    candidate_from_words,
    words_from_candidate,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The piece that is the convention; every other piece is a focus tag's criteria.
CONVENTION_PIECE = "convention"
# The usage of an iteration that a history line records, by key: this iteration's, not the sum.
ITERATION_USAGE_KEYS: tuple[str, ...] = (
    "jev_cost_usd",
    "luna_cost_usd",
    "luna_input_tokens",
    "luna_output_tokens",
)


@dataclasses.dataclass(frozen=True)
class SearchData:
    """The problems a search reads: the learning folds and the selection fold, never fold 0.

    Attributes:
        raw: The raw dev rows of the searched folds, with ``fold``, indexed by ``sample_id``.
        problems: ``raw_to_problems(raw)``: the rows the reflective records quote.
        labels: The 0/1 focus labels of ``problems``, one column per focus tag.
        learning_ids: The problems of the learning folds, ascending: GEPA's trainset.
        selection_ids: The problems of the selection fold, ascending: GEPA's valset.
        weights: The balanced Brier weights of the tags (``objective.tag_weights``), from the
            prevalences of the learning folds.
    """

    raw: pd.DataFrame
    problems: pd.DataFrame
    labels: pd.DataFrame
    learning_ids: list[int]
    selection_ids: list[int]
    weights: pd.DataFrame


def search_data(raw_dev: pd.DataFrame) -> SearchData:
    """Keep the dev problems a search reads, and weigh the tags on its learning folds.

    Args:
        raw_dev: The raw dev problems with their ``fold`` column, as
            ``select_part(read_samples(), split, "dev")`` gives them.

    Returns:
        The search's data; the rows of the held-out fold are dropped before anything reads them.

    Raises:
        ValueError: If ``raw_dev`` has no ``fold`` column, or the learning or selection folds
            hold no problem; as ``raw_to_problems`` and ``objective.tag_weights``.
    """
    if "fold" not in raw_dev:
        raise ValueError(
            "The dev problems have no fold column: pass select_part(raw, split, 'dev')'s rows."
        )
    raw = raw_dev[raw_dev["fold"].isin(SEARCH_FOLDS)]
    learning = sorted(int(i) for i in raw.index[raw["fold"].isin(SEARCH_LEARNING_FOLDS)])
    selection = sorted(int(i) for i in raw.index[raw["fold"] == SEARCH_SELECTION_FOLD])
    if not learning or not selection:
        raise ValueError(
            f"The learning folds {list(SEARCH_LEARNING_FOLDS)} hold {len(learning)} problems and "
            f"the selection fold {SEARCH_SELECTION_FOLD} holds {len(selection)}: a search needs "
            "both."
        )
    problems = raw_to_problems(raw)
    labels = focus_labels(problems)
    return SearchData(
        raw=raw,
        problems=problems,
        labels=labels,
        learning_ids=learning,
        selection_ids=selection,
        weights=tag_weights(labels.loc[learning]),
    )


def search_folder(root: Path = EXPERIMENTS_DIR) -> Path:
    """Return the folder of a search's caches and record: gitignored, under ``experiments/``.

    Args:
        root: The experiments folder.

    Returns:
        ``root / GEPA_NAME / SEARCH_FOLDER``; it may not exist yet.
    """
    return Path(root) / GEPA_NAME / SEARCH_FOLDER


class SearchLog(Protocol):
    """What a live log of the search is told, line by line (``search_tracking.LiveSearchLog``)."""

    def iteration(self, line: Mapping[str, object]) -> None:
        """Log one line of ``history.jsonl``, once its iteration is over."""

    def candidate(self, line: Mapping[str, object]) -> None:
        """Log one line of ``candidates.jsonl``, once its candidate is scored."""


class _Escape(BaseException):
    """Carries an error past GEPA, whose proposer catches every ``Exception`` and goes on."""

    def __init__(self, error: Exception) -> None:
        """Hold the error that ``run_search`` raises again."""
        super().__init__(error)
        self.error = error


class _QuietLogger:  # pylint: disable=too-few-public-methods
    """GEPA's logger, silenced: the recorder prints one line per iteration instead."""

    def log(self, message: str) -> None:
        """Drop one of GEPA's messages, which may quote a whole piece."""
        del message


def _kept(handler: Callable[[SearchRecorder, Any], None]) -> Callable[[SearchRecorder, Any], None]:
    """Wrap a callback of the recorder so that its error is kept, not dropped by GEPA."""

    @functools.wraps(handler)
    def wrapper(recorder: SearchRecorder, event: Any) -> None:
        try:
            handler(recorder, event)
        except Exception as error:  # pylint: disable=broad-exception-caught
            recorder.keep(error)

    return wrapper


class SearchRecorder:  # pylint: disable=too-many-instance-attributes
    """GEPA's callback that writes the search's record as it goes.

    ``history.jsonl`` gets one line per iteration, written once the next iteration starts or
    the search stops at a cap: an iteration that an error interrupted leaves no line, and the
    run that resumes the search writes it. ``candidates.jsonl`` gets one line per candidate,
    once GEPA has scored it on the selection fold; the seed is candidate 0, at iteration 0.
    Both files are written from the start on every run. ``finish`` writes ``outcome.json``.

    GEPA logs and drops an error raised in a callback: the recorder keeps the first one, which
    the adapter raises at its next evaluation (``raise_kept``), and ``run_search`` at the end;
    a later one is named on stderr.

    Attributes:
        folder: The search's folder.
        data: The search's problems.
        cache: The Jev cache: what Jev's answers cost so far, and the selection fold's
            probabilities of a candidate, which GEPA does not keep for the seed.
        live: The live log of the search, or None.
        iteration: The number of the iteration under way, from 1; None before the first.
        error: The first error a callback raised, or None.
    """

    def __init__(
        self, folder: Path, data: SearchData, cache: JevCache, live: SearchLog | None = None
    ) -> None:
        """Hold where the record goes; nothing is written before the search starts.

        Args:
            folder: The search's folder (``search_folder``).
            data: The search's problems.
            cache: The search's Jev cache.
            live: The live log of the search; None for none.
        """
        self.folder = Path(folder)
        self.data = data
        self.cache = cache
        self.live = live
        self.iteration: int | None = None
        self.error: Exception | None = None
        self._line: dict[str, Any] = {}
        self._pending: dict[str, Any] | None = None
        self._usage_at_start: dict[str, float] = {}
        self._candidates: dict[int, dict[str, str]] = {}
        self._reflections: dict[str, LlmAnswer] = {}

    def costs(self) -> dict[str, float]:
        """Return what the search's answers cost so far: Jev's, the reflection model's, both.

        Returns:
            ``{"jev", "luna", "total"}`` in USD, each answer counted once, the seed's excluded.
        """
        usage = self._usage()
        jev, luna = usage["jev_cost_usd"], usage["luna_cost_usd"]
        return {"jev": jev, "luna": luna, "total": jev + luna}

    def keep(self, error: Exception) -> None:
        """Keep the first error a callback raised; a later one is only named on stderr.

        Args:
            error: The error a callback raised.
        """
        if self.error is None:
            self.error = error
            return
        print(
            f"The search's record failed again, and this error is not kept: "
            f"{type(error).__name__}: {error} (the first, a {type(self.error).__name__}, stops "
            "the search).",
            file=sys.stderr,
            flush=True,
        )

    def raise_kept(self) -> None:
        """Raise the error a callback kept, if any.

        Raises:
            Exception: The first error a callback raised.
        """
        if self.error is not None:
            raise self.error

    def reflected(self, answer: LlmAnswer) -> None:
        """Count an answer of the reflection model: its cost and tokens, once per request."""
        self._reflections[answer.request_sha256] = answer

    def refused(self, piece: str, reason: str) -> None:
        """Record that the rewrite of a piece was refused, and why (``reflection``)."""
        del piece
        self._line["refusal"] = reason

    @_kept
    def on_optimization_start(self, event: Mapping[str, Any]) -> None:
        """Start the record afresh: empty both files of lines, and remove the outcome."""
        del event
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in (HISTORY_LINES_FILE, CANDIDATE_LINES_FILE):
            (self.folder / name).write_text("", encoding="utf-8")
        (self.folder / OUTCOME_FILE).unlink(missing_ok=True)

    @_kept
    def on_iteration_start(self, event: Mapping[str, Any]) -> None:
        """Write the line of the iteration before, then start this iteration's."""
        self._flush()
        self.iteration = int(event["iteration"])
        self._line = _new_line(self.iteration)
        self._usage_at_start = self._usage()

    @_kept
    def on_candidate_selected(self, event: Mapping[str, Any]) -> None:
        """Record the parent candidate of the iteration."""
        self._line["parent"] = int(event["candidate_idx"])

    @_kept
    def on_minibatch_sampled(self, event: Mapping[str, Any]) -> None:
        """Record the minibatch, as sample ids (GEPA's ids are positions in the trainset)."""
        self._line["minibatch"] = [self.data.learning_ids[i] for i in event["minibatch_ids"]]

    @_kept
    def on_evaluation_end(self, event: Mapping[str, Any]) -> None:
        """Record the sum of the minibatch scores of the parent, or of the child."""
        key = "child_sum" if event["candidate_idx"] is None else "parent_sum"
        self._line[key] = math.fsum(event["scores"])

    @_kept
    def on_reflective_dataset_built(self, event: Mapping[str, Any]) -> None:
        """Record the piece the iteration rewrites."""
        self._line["piece"] = event["components"][0]

    @_kept
    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        """Record that the child was accepted, and its index."""
        self._line["accepted"] = True
        self._line["candidate"] = int(event["new_candidate_idx"])

    @_kept
    def on_valset_evaluated(self, event: Mapping[str, Any]) -> None:
        """Write the line of a candidate scored on the selection fold, and log it."""
        line = self._candidate_line(event)
        self._candidates[line["index"]] = dict(event["candidate"])
        _append_line(self.folder / CANDIDATE_LINES_FILE, line)
        if self.live is not None:
            self.live.candidate(line)

    @_kept
    def on_iteration_end(self, event: Mapping[str, Any]) -> None:
        """Close the iteration's line with its own cost and tokens; it is written later."""
        del event
        usage = self._usage()
        for key in ITERATION_USAGE_KEYS:
            self._line[key] = usage[key] - self._usage_at_start[key]
        self._pending = self._line

    def finish(self, outcome: SearchOutcome, *, max_cost_usd: float, max_iterations: int) -> None:
        """Write the last iteration's line, then ``outcome.json``: the search stopped at a cap.

        Args:
            outcome: How the search ended.
            max_cost_usd: The cost cap this run used, in USD.
            max_iterations: The iteration cap this run used.
        """
        self._flush()
        values = {
            **dataclasses.asdict(outcome),
            "max_cost_usd": max_cost_usd,
            "max_iterations": max_iterations,
            "gepa_version": importlib.metadata.version("gepa"),
        }
        text = json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False)
        (self.folder / OUTCOME_FILE).write_text(f"{text}\n", encoding="utf-8")

    def _usage(self) -> dict[str, float]:
        """Return the cost and tokens of every answer used so far, by ``ITERATION_USAGE_KEYS``."""
        answers = list(self._reflections.values())
        return {
            "jev_cost_usd": self.cache.used_usd(),
            "luna_cost_usd": math.fsum(answer_cost_usd(a)["total_cost"] for a in answers),
            "luna_input_tokens": sum(answer.usage["input_tokens"] for answer in answers),
            "luna_output_tokens": sum(answer.usage["output_tokens"] for answer in answers),
        }

    def _flush(self) -> None:
        """Write the line of the last iteration that ended, log it and print its progress."""
        if self._pending is None:
            return
        line, self._pending = self._pending, None
        _append_line(self.folder / HISTORY_LINES_FILE, line)
        if self.live is not None:
            self.live.iteration(line)
        print(_progress(line, self.costs()["total"]), flush=True)

    def _candidate_line(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Describe a candidate scored on the selection fold: words, scores, balanced Brier."""
        words = words_from_candidate(event["candidate"])
        selection = self.data.selection_ids
        scores = {
            str(selection[position]): float(score)
            for position, score in sorted(event["scores_by_val_id"].items())
        }
        parents = [int(parent) for parent in event["parent_ids"] if parent is not None]
        brier = balanced_brier(
            self.cache.probabilities(words, selection), self.data.labels.loc[selection]
        )
        return {
            "index": int(event["candidate_idx"]),
            "parents": parents,
            "iteration": int(event["iteration"]),
            "piece": self._changed_piece(event["candidate"], parents),
            "words": words.settings(),
            "words_sha256": words.sha256(),
            "selection_scores": scores,
            "selection_mean": math.fsum(scores.values()) / len(scores),
            "selection_brier": {tag: float(brier[tag]) for tag in FOCUS_TAGS},
        }

    def _changed_piece(self, candidate: Mapping[str, str], parents: Sequence[int]) -> str | None:
        """Name the piece a candidate changed from its parent; None for the seed."""
        if not parents:
            return None
        parent = self._candidates[parents[0]]
        changed = [piece for piece in PIECES if candidate[piece] != parent[piece]]
        return changed[0] if changed else None


class JevSearchAdapter:
    """GEPA's adapter of the search: Jev scores a candidate, the reflection model rewrites it.

    It follows GEPA's ``GEPAAdapter`` protocol. A data instance is a sample id; a candidate is
    ``{piece: text}`` (``words.candidate_from_words``).

    Attributes:
        data: The search's problems.
        cache: Jev's probabilities, each answer asked once.
        reflector: The reflection model's answers, each asked once (``reflection.Reflector``).
        recorder: The search's recorder: the iteration under way, and what it records.
    """

    def __init__(
        self, data: SearchData, cache: JevCache, reflector: Reflector, recorder: SearchRecorder
    ) -> None:
        """Hold the parts of the search.

        Args:
            data: The search's problems.
            cache: The search's Jev cache.
            reflector: The search's reflection model.
            recorder: The search's recorder.
        """
        self.data = data
        self.cache = cache
        self.reflector = reflector
        self.recorder = recorder
        self._searched = frozenset(data.learning_ids) | frozenset(data.selection_ids)

    def evaluate(
        self, batch: list[int], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        """Score a candidate on some problems: one minus each problem's balanced Brier term.

        Args:
            batch: Sample ids of the learning or selection folds.
            candidate: The candidate's pieces.
            capture_traces: Whether to give each problem's trajectory, for reflection.

        Returns:
            GEPA's ``EvaluationBatch``: each problem's probabilities (``outputs``), its score
            ``s_i`` (``objective.problem_scores``), each tag's ``1 - w_t(y) (p - y)^2``
            (``objective_scores``) and, with ``capture_traces``, ``{"sample_id": id}``.

        Raises:
            ValueError: If a problem is not of the learning or selection folds: the search never
                reads the held-out fold.
            SearchInterrupted: As ``JevCache.probabilities``.
            Exception: The first error a callback of the recorder raised.
        """
        self.recorder.raise_kept()
        ids = [int(sample_id) for sample_id in batch]
        outside = [sample_id for sample_id in ids if sample_id not in self._searched]
        if outside:
            raise ValueError(
                f"The problems {problem_list(outside)} are not of the learning or selection "
                f"folds: the search never reads the held-out fold (fold {SEARCH_HELD_OUT_FOLD})."
            )
        probabilities = self.cache.probabilities(words_from_candidate(candidate), ids)
        labels = self.data.labels.loc[ids]
        terms = 1.0 - weighted_errors(probabilities, labels, self.data.weights)
        scores = problem_scores(probabilities, labels, self.data.weights)
        return EvaluationBatch(
            outputs=_rows(probabilities),
            scores=[float(score) for score in scores],
            trajectories=[{"sample_id": i} for i in ids] if capture_traces else None,
            objective_scores=_rows(terms),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build each piece's records from an evaluated minibatch (``reflection``).

        Args:
            candidate: The parent's pieces; the records read only the evaluation.
            eval_batch: The parent's evaluation, with trajectories.
            components_to_update: The pieces to rewrite.

        Returns:
            The records of each piece: ``tag_records`` for a tag, ``convention_records`` for the
            convention.

        Raises:
            _Escape: Carrying any error, which GEPA would otherwise drop.
        """
        del candidate
        try:
            return self._records(eval_batch, components_to_update)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise _Escape(error) from error

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Ask the reflection model to rewrite each piece, and keep the rewrites that pass.

        A refused rewrite (``reflection.read_proposal``) is recorded and left out, so the piece
        keeps the parent's text; with no rewrite left, GEPA skips the child.

        Args:
            candidate: The parent's pieces.
            reflective_dataset: The records of each piece (``make_reflective_dataset``).
            components_to_update: The pieces to rewrite.

        Returns:
            The new text of each piece whose rewrite passed.

        Raises:
            _Escape: Carrying any error (e.g. a ``TransientApiError`` of the reflection model),
                which GEPA would otherwise drop, asking again or skipping the child.
        """
        try:
            return self._propose(candidate, reflective_dataset, components_to_update)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise _Escape(error) from error

    def _records(
        self, eval_batch: EvaluationBatch, pieces: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Build each piece's records from an evaluation's trajectories and outputs."""
        ids = [int(trajectory["sample_id"]) for trajectory in eval_batch.trajectories]
        index = pd.Index(ids, name="sample_id")
        probabilities = pd.DataFrame(eval_batch.outputs, index=index, columns=list(FOCUS_TAGS))
        problems = self.data.problems.loc[ids]
        labels = self.data.labels.loc[ids]
        weights = self.data.weights
        return {
            piece: (
                convention_records(problems, probabilities, labels, weights)
                if piece == CONVENTION_PIECE
                else tag_records(piece, problems, probabilities, labels, weights)
            )
            for piece in pieces
        }

    def _propose(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        pieces: Sequence[str],
    ) -> dict[str, str]:
        """Ask for each piece's rewrite and keep those that pass; record the refused ones."""
        words = words_from_candidate(candidate)
        texts = {}
        for piece in pieces:
            records = reflective_dataset[piece]
            request = reflection_request(piece, words, records, self.recorder.iteration)
            answer = self.reflector.answer(request)
            self.recorder.reflected(answer)
            proposal = read_proposal(piece, answer, texts_shown(records))
            if proposal.text is None:
                self.recorder.refused(piece, proposal.reason)
            else:
                texts[piece] = proposal.text
        return texts


class IterationCap:  # pylint: disable=too-few-public-methods
    """GEPA's stopper at a number of iterations, checked before each one.

    Attributes:
        max_iterations: The iterations the search may run.
        iterations: The iterations done when last checked.
        reached: Whether the last check stopped the search.
    """

    def __init__(self, max_iterations: int) -> None:
        """Hold the cap.

        Args:
            max_iterations: The iterations the search may run, at least 1.

        Raises:
            ValueError: If ``max_iterations`` is below 1.
        """
        if max_iterations < 1:
            raise ValueError(f"A search runs at least 1 iteration, not {max_iterations}.")
        self.max_iterations = max_iterations
        self.iterations = 0
        self.reached = False

    def __call__(self, state: Any) -> bool:
        """Tell whether the search has run its iterations.

        Args:
            state: GEPA's state; ``state.i + 1`` iterations are done.

        Returns:
            Whether ``max_iterations`` are done.
        """
        self.iterations = state.i + 1
        self.reached = self.iterations >= self.max_iterations
        return self.reached


class CostCap:  # pylint: disable=too-few-public-methods
    """GEPA's stopper at a cost, checked before each iteration.

    The iteration under way when the cap is crossed finishes, so the search may go over the cap
    by one iteration's cost.

    Attributes:
        spent: Returns what the search's answers cost so far, in USD.
        max_cost_usd: The cap, in USD.
        reached: Whether the last check stopped the search.
    """

    def __init__(self, spent: Callable[[], float], max_cost_usd: float) -> None:
        """Hold the cap.

        Args:
            spent: Returns what the search's answers cost so far, in USD.
            max_cost_usd: The cap, in USD, above 0.

        Raises:
            ValueError: If ``max_cost_usd`` is not above 0 (NaN included).
        """
        if not max_cost_usd > 0:
            raise ValueError(f"A cost cap must be above 0 USD, not {max_cost_usd}.")
        self.spent = spent
        self.max_cost_usd = max_cost_usd
        self.reached = False

    def __call__(self, state: Any) -> bool:
        """Tell whether the search's answers cost the cap.

        Args:
            state: GEPA's state, unused.

        Returns:
            Whether ``spent()`` is at least ``max_cost_usd``.
        """
        del state
        self.reached = self.spent() >= self.max_cost_usd
        return self.reached


@dataclasses.dataclass(frozen=True)
class SearchOutcome:
    """How a search ended, at a cap.

    Attributes:
        stop_reason: ``"iterations"`` or ``"cost"``: the cap that stopped it.
        iterations: The iterations it ran.
        best_index: GEPA's best candidate, the highest mean score on the selection fold, before
            the dataset-text check that ``export`` runs.
        cost_usd: What its answers cost: ``{"jev", "luna", "total"}``, in USD.
    """

    stop_reason: str
    iterations: int
    best_index: int
    cost_usd: dict[str, float]


def run_search(  # pylint: disable=too-many-arguments
    data: SearchData,
    folder: Path,
    *,
    max_cost_usd: float,
    max_iterations: int = SEARCH_MAX_ITERATIONS,
    committed_seed_answers: Sequence[LlmAnswer],
    jev_session: JevSession | None = None,
    reflection_ask: Ask | None = None,
    live: SearchLog | None = None,
) -> SearchOutcome:
    """Run the search, or replay it from its caches and go on, until a cap stops it.

    Args:
        data: The search's problems (``search_data``).
        folder: The search's folder (``search_folder``): ``jev/``, ``reflections.jsonl``,
            ``history.jsonl``, ``candidates.jsonl`` and ``outcome.json``.
        max_cost_usd: The cap on what Jev's and the reflection model's answers cost, in USD.
        max_iterations: The cap on the iterations.
        committed_seed_answers: The committed dev answers of ``jev-statement-code``, which
            score the seed.
        jev_session: The Jev session to send through; None for ``client.jev_session``.
        reflection_ask: The reflection model's ``Ask``; None for the toolkit's OpenAI adapter.
        live: The live log of the search; None for none.

    Returns:
        How the search ended; ``outcome.json`` holds it too, with the caps this run used.

    Raises:
        SearchInterrupted: If Jev or the reflection model left a request unanswered: run the
            search again to resume it; ``CostCapSpent`` if Jev's answers spent the cap inside a
            call: run it again with a higher cap.
        ValueError: If a cap is not positive, or as ``JevCache.prefill_seed``.
        Exception: Any other error of the search, which GEPA would otherwise drop.
    """
    iteration_cap = IterationCap(max_iterations)
    folder = Path(folder)
    cache = JevCache(
        folder / JEV_CACHE_FOLDER, data.raw, max_cost_usd=max_cost_usd, jev_session=jev_session
    )
    # The seed's answers are on disk before GEPA scores it: no seed request is ever sent.
    cache.prefill_seed(committed_seed_answers)
    recorder = SearchRecorder(folder, data, cache, live)
    cost_cap = CostCap(lambda: recorder.costs()["total"], max_cost_usd)
    reflector = Reflector(folder / REFLECTIONS_FILE, ask=reflection_ask)
    result = _optimize(
        data, JevSearchAdapter(data, cache, reflector, recorder), (iteration_cap, cost_cap)
    )
    recorder.raise_kept()
    outcome = SearchOutcome(
        stop_reason="iterations" if iteration_cap.reached else "cost",
        iterations=iteration_cap.iterations,
        best_index=result.best_idx,
        cost_usd=recorder.costs(),
    )
    recorder.finish(outcome, max_cost_usd=max_cost_usd, max_iterations=max_iterations)
    return outcome


def _optimize(
    data: SearchData, adapter: JevSearchAdapter, stoppers: Sequence[Callable[[Any], bool]]
) -> Any:
    """Run ``gepa.optimize`` from the seed; raise again an error GEPA would have dropped."""
    sampler = StratifiedSampler(
        data.learning_ids,
        data.labels,
        size=SEARCH_MINIBATCH_SIZE,
        per_tag=SEARCH_POSITIVES_PER_TAG,
        seed=SEARCH_SEED,
    )
    try:
        return gepa.optimize(
            seed_candidate=candidate_from_words(SEED_WORDS),
            trainset=list(data.learning_ids),
            valset=list(data.selection_ids),
            adapter=adapter,
            batch_sampler=sampler,
            stop_callbacks=list(stoppers),
            callbacks=[adapter.recorder],
            logger=_QuietLogger(),
            **SEARCH_GEPA_OPTIONS,
        )
    except _Escape as escape:
        if isinstance(escape.error, TransientApiError):
            raise SearchInterrupted(
                f"The reflection model did not answer at iteration {adapter.recorder.iteration} "
                f"({type(escape.error).__name__}): run the search again to resume it."
            ) from escape.error
        raise escape.error from None


def _rows(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Return each row of a frame of focus tags as ``{tag: value}``, in Python floats."""
    return [
        {tag: float(value) for tag, value in zip(FOCUS_TAGS, row, strict=True)}
        for row in frame[list(FOCUS_TAGS)].itertuples(index=False)
    ]


def _new_line(iteration: int) -> dict[str, Any]:
    """Return an iteration's history line before anything happened, keys in their order."""
    return {
        "iteration": iteration,
        "parent": None,
        "piece": None,
        "minibatch": [],
        "parent_sum": None,
        "child_sum": None,
        "accepted": False,
        "refusal": None,
        "candidate": None,
        "jev_cost_usd": 0.0,
        "luna_cost_usd": 0.0,
        "luna_input_tokens": 0,
        "luna_output_tokens": 0,
    }


def _append_line(path: Path, line: Mapping[str, Any]) -> None:
    """Append one JSON line to a file of the record."""
    text = json.dumps(line, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{text}\n")


def _progress(line: Mapping[str, Any], total_usd: float) -> str:
    """Describe an iteration in one line: its piece, its parent, what became of the child."""
    if line["refusal"] is not None:
        verdict = f"refused ({line['refusal']})"
    elif line["accepted"]:
        verdict = f"accepted as candidate {line['candidate']}"
    else:
        verdict = "rejected"
    return (
        f"iteration {line['iteration']}: {line['piece']} of candidate {line['parent']}, "
        f"{verdict}; ${total_usd:.4f} spent"
    )
