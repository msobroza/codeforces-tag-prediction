"""The search in MLflow: logged live by ``tune``, or replayed from ``tuning/`` (docs/adr/0010).

The search logs into its own experiment, ``SEARCH_EXPERIMENT`` (``jev-gepa``), of the store that
``use_store`` opens, and never into ``tag-methods``:

- one parent run, ``SEARCH_RUN_NAME``, tagged ``search`` with the searched method's name. Its
  parameters are the search settings, flattened (``reflection.model``, ``gepa.seed``). At each
  iteration (step = iteration, gepa's 1-based number) it gets the metrics
  ``selection_best_score`` and ``selection_brier.<tag>`` (of the best candidate scored so far,
  among every candidate: the live log cannot know the export's dataset-text check, so a candidate
  it later refuses counts too, and ``log_search`` logs the same), ``minibatch_parent_sum`` and
  ``minibatch_child_sum`` (each absent when that candidate was not scored on the minibatch),
  ``accepted``, and the cumulative ``cost_usd_jev``, ``cost_usd_luna`` and ``cost_usd_total``;
- one nested run per candidate scored on the selection fold, ``candidate-<index>``: its parents,
  iteration and piece as parameters, its ``selection_mean`` and ``selection_brier.<tag>``, and
  the tags ``candidate`` and ``words_sha256``;
- one MLflow prompt per piece of Jev's words, ``prompt_name(piece)``. A candidate adds a version
  of each piece whose text the prompt does not hold yet: the seed adds version 1 of all nine,
  and any other candidate a version of the piece it changed. Each version is tagged
  ``content_sha256`` (of its text), ``candidate`` and ``iteration``, and linked to the run of the
  candidate that added it and of each candidate whose changed piece holds its text (a candidate
  whose words another one already holds adds no version, and links the existing one).

``LiveSearchLog`` logs as ``tune``'s recorder notifies it; ``log_search`` replays the frozen
record into a store, e.g. an empty one rebuilt from committed files. Both log only what the store
lacks: a step it holds, a candidate's run, a version of the same text are skipped. So a resumed
search adds its new iterations only, and replaying after the live log, or twice, adds nothing. A
prompt holding a version with no ``content_sha256`` tag (made by hand, or by other code) is
refused: none of its texts can be matched, so the store must be rebuilt.
``log_search`` also writes what only the export knows: each candidate's ``passed_check`` tag,
each prompt's ``chosen`` alias (the version of the chosen words' piece), the parent's ``chosen``
tag, and the artifacts ``candidates.csv`` and ``history.csv``. The words of a candidate that
failed the dataset-text check are not in ``tuning/``, so a replayed store holds no version of
them; a live store may (it is local and gitignored, and the screenshots come from a replayed one).

Versions are made with ``MlflowClient.create_prompt_version``, never ``register_prompt``, and
never formatted (docs/adr/0002, safeguard 4). mlflow is imported only inside the functions that
log, once ``use_store`` has set MLflow's environment: the package ``typesafe_jev`` sets none of
it, so a module-level import here could let MLflow start its telemetry or its hint to coding
agents in a command that imports this module first.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.questions import question_id
from codeforces_tags.methods.typesafe_jev.search_record import SearchRecord, parents_of
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    SEARCH_EXPERIMENT,
    SEARCH_PROMPT_PREFIX,
    SEARCH_RUN_NAME,
    SEARCH_SETTINGS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    JevWords,
    piece_text,
    words_from_json,
)
from codeforces_tags.tracking.environment import use_store

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Run
    from mlflow.entities.model_registry import PromptVersion

# The parent run's tag, whose value is the searched method's name: how a store's search is found.
SEARCH_TAG = "search"
# The tags of a candidate's run and of a prompt version.
CANDIDATE_TAG = "candidate"
ITERATION_TAG = "iteration"
WORDS_TAG = "words_sha256"
CHECK_TAG = "passed_check"
CONTENT_TAG = "content_sha256"
# The parent run's tag and each prompt's alias that name the chosen candidate (log_search only).
CHOSEN = "chosen"
# The parent run's metric logged at every step: the steps a store holds are its steps.
STEP_METRIC = "cost_usd_total"
# MLflow's tag of a nested run's parent.
PARENT_RUN_TAG = "mlflow.parentRunId"
# How many versions of a prompt one search_prompt_versions call reads (MLflow's own default page);
# every page is read.
VERSIONS_PAGE_SIZE = 100
# How many hex digits of a hash an error message shows.
_SHORT_HASH = 12


@dataclass(frozen=True)
class _Candidate:  # pylint: disable=too-many-instance-attributes
    """One candidate scored on the selection fold, read from a live line or a frozen row."""

    index: int
    parents: tuple[int, ...]
    iteration: int
    piece: str | None  # the piece it changed; None for the seed
    selection_mean: float
    selection_brier: dict[str, float]  # by focus tag, in FOCUS_TAGS order
    words_sha256: str
    words: JevWords | None  # None when tuning/ leaves its words out
    passed_check: bool | None  # None in a live log: the export checks it


@dataclass(frozen=True)
class _Step:
    """One iteration of the search, read from a live line or a frozen row."""

    iteration: int
    parent_sum: float | None  # None when the parent was not scored on the minibatch
    child_sum: float | None  # None when no child was scored (a refused proposal)
    accepted: bool
    jev_cost_usd: float
    luna_cost_usd: float


def prompt_name(piece: str) -> str:
    """Return the MLflow prompt of a piece of Jev's words, its spaces as dashes.

    Args:
        piece: ``"convention"`` or a focus tag, e.g. ``"number theory"``.

    Returns:
        ``SEARCH_PROMPT_PREFIX`` and the piece, e.g. ``"jev-gepa-number-theory"``.
    """
    return SEARCH_PROMPT_PREFIX + piece.replace(" ", "-")


def metric_suffix(tag: str) -> str:
    """Return a focus tag as it ends a metric's name: its question id, spaces as underscores.

    Args:
        tag: A focus tag, e.g. ``"number theory"``.

    Returns:
        ``questions.question_id(tag)``, e.g. ``"number_theory"``
        (``selection_brier.number_theory``).
    """
    return question_id(tag)


class LiveSearchLog:
    """The search's log in MLflow, written as ``tune``'s recorder notifies it.

    The recorder logs each candidate scored in an iteration before that iteration's line, and
    the seed (iteration 0) first: an iteration's ``selection_best_score`` is the best of the
    candidates logged so far whose iteration is at most its own, whether or not the export's
    dataset-text check passes them later.
    """

    def __init__(self, *, tracking_uri: str, artifact_dir: Path, run_file: Path) -> None:
        """Open the store's search run, created if missing, and write its id to ``run_file``.

        Args:
            tracking_uri: The MLflow store, a ``sqlite:///`` URI.
            artifact_dir: The artifact folder of a new experiment.
            run_file: The search's ``run.json``: the run's id, its experiment and its store.

        Raises:
            ValueError: As ``use_store`` does; if the experiment holds two search runs; or if
                its search run holds other search settings than ``SEARCH_SETTINGS``.
        """
        self._run = _SearchRun.open(SEARCH_SETTINGS, tracking_uri, artifact_dir)
        self._run.write_run_file(run_file)
        self._candidates: list[_Candidate] = []
        self._steps: list[_Step] = []

    @property
    def run_id(self) -> str:
        """The id of the search's parent run."""
        return self._run.parent_id

    def candidate(self, line: Mapping[str, object]) -> None:
        """Log a candidate scored on the selection fold: its run and its pieces' new versions.

        Args:
            line: Its line of ``candidates.jsonl``.

        Raises:
            ValueError: If the store holds another candidate's words under its index, or a prompt
                of a piece holds a version with no ``content_sha256`` tag.
        """
        candidate = _live_candidate(line)
        self._candidates.append(candidate)
        self._run.log_candidate(candidate)

    def iteration(self, line: Mapping[str, object]) -> None:
        """Log an iteration's metrics at its step, unless the store holds that step.

        Args:
            line: Its line of ``history.jsonl``, notified once its candidates were.
        """
        self._steps.append(_live_step(line))
        self._run.log_step(self._steps, self._candidates)

    def close(self) -> None:
        """Mark the search run finished; a resumed search logs into it again."""
        self._run.finish()


def log_search(
    record: SearchRecord, *, tracking_uri: str, artifact_dir: Path, run_file: Path | None = None
) -> str:
    """Replay a frozen search into a store, logging only what the store lacks.

    It needs no API key and sends no request: everything comes from ``tuning/``.

    Args:
        record: The frozen record, ``read_search_record(tuning)``.
        tracking_uri: The MLflow store, a ``sqlite:///`` URI.
        artifact_dir: The artifact folder of a new experiment.
        run_file: Where to write the parent run's id, its experiment and its store, if given.

    Returns:
        The id of the search's parent run.

    Raises:
        ValueError: As ``use_store`` does; if the experiment holds two search runs; if its
            search run holds other search settings than the record's; if it holds another
            candidate's words under an index of the record; or if a prompt of a piece holds a
            version with no ``content_sha256`` tag.
    """
    run = _SearchRun.open(record.params["search_settings"], tracking_uri, artifact_dir)
    candidates = _record_candidates(record)
    for candidate in candidates:
        run.log_candidate(candidate)
    steps = _record_steps(record.history)
    for end in range(1, len(steps) + 1):
        run.log_step(steps[:end], candidates)
    chosen = int(record.params["chosen"])
    run.mark_chosen(chosen, record.words[chosen])
    run.log_tables(record)
    run.finish()
    if run_file is not None:
        run.write_run_file(run_file)
    return run.parent_id


class _SearchRun:
    """A store's search run, its candidates' runs and its prompts, with what they already hold."""

    def __init__(self, client: MlflowClient, store: tuple[str, str], parent_id: str) -> None:
        self.client = client
        self.tracking_uri, self.experiment_id = store
        self.parent_id = parent_id
        history = client.get_metric_history(parent_id, STEP_METRIC)
        self.logged_steps = {metric.step for metric in history}
        children = client.search_runs(
            [self.experiment_id], filter_string=f"tags.`{PARENT_RUN_TAG}` = '{parent_id}'"
        )
        self.candidate_runs: dict[int, Run] = {
            int(child.data.tags[CANDIDATE_TAG]): child for child in children
        }
        self.versions: dict[str, dict[str, PromptVersion]] = {}

    @classmethod
    def open(
        cls, settings: Mapping[str, object], tracking_uri: str, artifact_dir: Path
    ) -> _SearchRun:
        """Find the experiment's search run, or create it with the settings as parameters."""
        experiment_id = use_store(tracking_uri, SEARCH_EXPERIMENT, artifact_dir)
        from mlflow import MlflowClient  # pylint: disable=import-outside-toplevel

        client = MlflowClient()
        params = _flat_params(settings)
        found = client.search_runs(
            [experiment_id], filter_string=f"tags.{SEARCH_TAG} = '{GEPA_NAME}'"
        )
        if len(found) > 1:
            raise ValueError(
                f"The experiment {SEARCH_EXPERIMENT} of {tracking_uri} holds {len(found)} search "
                f"runs of {GEPA_NAME}, not one: rebuild the store (notebooks/README.md, Figures)."
            )
        if found:
            _check_params(found[0], params, tracking_uri)
            return cls(client, (tracking_uri, experiment_id), found[0].info.run_id)
        run = client.create_run(
            experiment_id, run_name=SEARCH_RUN_NAME, tags={SEARCH_TAG: GEPA_NAME}
        )
        _log_batch(client, run.info.run_id, params=params)
        return cls(client, (tracking_uri, experiment_id), run.info.run_id)

    def write_run_file(self, run_file: Path) -> None:
        """Write the parent run's id, its experiment and its store to ``run_file``."""
        values = {
            "run_id": self.parent_id,
            "experiment_id": self.experiment_id,
            "tracking_uri": self.tracking_uri,
        }
        run_file.parent.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

    def log_candidate(self, candidate: _Candidate) -> None:
        """Log a candidate's run once, its check when known, and its pieces' new versions."""
        run_id = self._candidate_run(candidate)
        if candidate.passed_check is not None:
            self.client.set_tag(run_id, CHECK_TAG, str(candidate.passed_check).lower())
        if candidate.words is not None:
            self._log_versions(candidate, run_id)

    def _candidate_run(self, candidate: _Candidate) -> str:
        """Return the candidate's run, created with its scores, refusing another's words."""
        found = self.candidate_runs.get(candidate.index)
        if found is not None:
            logged = found.data.tags.get(WORDS_TAG, "")
            if logged != candidate.words_sha256:
                raise ValueError(
                    f"The candidate {candidate.index} of the search run {self.parent_id} has the "
                    f"words {logged[:_SHORT_HASH]}, not {candidate.words_sha256[:_SHORT_HASH]}: "
                    f"{self.tracking_uri} holds another search of {GEPA_NAME}. Rebuild the "
                    "store (notebooks/README.md, Figures) to log this one."
                )
            return found.info.run_id
        tags = {
            PARENT_RUN_TAG: self.parent_id,
            CANDIDATE_TAG: str(candidate.index),
            WORDS_TAG: candidate.words_sha256,
        }
        run = self.client.create_run(
            self.experiment_id, run_name=f"candidate-{candidate.index}", tags=tags
        )
        run_id = run.info.run_id
        _log_batch(
            self.client, run_id, params=_candidate_params(candidate), metrics=_scores(candidate)
        )
        self.client.set_terminated(run_id)
        self.candidate_runs[candidate.index] = self.client.get_run(run_id)
        return run_id

    def _log_versions(self, candidate: _Candidate, run_id: str) -> None:
        """Add a version of each piece whose text is new; link the candidate's versions to it.

        A candidate's versions are those it added and the version of the piece it changed, which
        another candidate holding the same text may have added first.
        """
        for piece in PIECES:
            text = piece_text(candidate.words, piece)
            version = self.version_of(piece, text)
            if version is None:
                version = self._add_version(piece, text, candidate)
            if piece == candidate.piece or version.tags.get(CANDIDATE_TAG) == str(candidate.index):
                self.client.link_prompt_version_to_run(run_id, version)

    def _prompt_versions(self, piece: str) -> dict[str, PromptVersion]:
        """Return a piece's versions by content hash, creating its prompt when it is new."""
        if piece not in self.versions:
            name = prompt_name(piece)
            if self.client.get_prompt(name) is None:
                self.client.create_prompt(name)
            versions = _all_versions(self.client, name)
            _check_content_tags(name, versions, self.tracking_uri)
            self.versions[piece] = {version.tags[CONTENT_TAG]: version for version in versions}
        return self.versions[piece]

    def version_of(self, piece: str, text: str) -> PromptVersion | None:
        """Return the version of a piece that holds a text, if any."""
        return self._prompt_versions(piece).get(_text_sha256(text))

    def _add_version(self, piece: str, text: str, candidate: _Candidate) -> PromptVersion:
        """Create the next version of a piece's prompt, holding a candidate's text."""
        content = _text_sha256(text)
        version = self.client.create_prompt_version(
            name=prompt_name(piece),
            template=text,
            description=f"candidate {candidate.index}, iteration {candidate.iteration}",
            tags={
                CONTENT_TAG: content,
                CANDIDATE_TAG: str(candidate.index),
                ITERATION_TAG: str(candidate.iteration),
            },
        )
        self.versions[piece][content] = version
        return version

    def log_step(self, steps: Sequence[_Step], candidates: Sequence[_Candidate]) -> None:
        """Log the last step's metrics, unless the store holds that step."""
        step = steps[-1]
        if step.iteration in self.logged_steps:
            return
        metrics = _step_metrics(steps) | _best_metrics(step.iteration, candidates)
        _log_batch(self.client, self.parent_id, metrics=metrics, step=step.iteration)
        self.logged_steps.add(step.iteration)

    def mark_chosen(self, chosen: int, words: JevWords) -> None:
        """Alias each piece's version of the chosen words ``chosen``, then tag the parent run."""
        for piece in PIECES:
            version = self.version_of(piece, piece_text(words, piece))
            self.client.set_prompt_alias(prompt_name(piece), CHOSEN, version.version)
        self.client.set_tag(self.parent_id, CHOSEN, str(chosen))

    def log_tables(self, record: SearchRecord) -> None:
        """Log the record's candidates and history as the parent run's CSV artifacts."""
        self.client.log_text(self.parent_id, record.candidates.to_csv(), "candidates.csv")
        self.client.log_text(self.parent_id, record.history.to_csv(), "history.csv")

    def finish(self) -> None:
        """Mark the parent run finished."""
        self.client.set_terminated(self.parent_id)


def _candidate_params(candidate: _Candidate) -> dict[str, str]:
    """Return a candidate run's parameters; the seed's parents and piece are empty."""
    return {
        "parents": " ".join(str(parent) for parent in candidate.parents),
        "iteration": str(candidate.iteration),
        "piece": candidate.piece or "",
    }


def _scores(candidate: _Candidate) -> dict[str, float]:
    """Return a candidate's selection-fold mean score and per-tag balanced Brier scores."""
    brier = {
        f"selection_brier.{metric_suffix(tag)}": value
        for tag, value in candidate.selection_brier.items()
    }
    return {"selection_mean": candidate.selection_mean} | brier


def _step_metrics(steps: Sequence[_Step]) -> dict[str, float]:
    """Return the last step's minibatch sums and acceptance, and the costs up to it."""
    step = steps[-1]
    jev = sum(each.jev_cost_usd for each in steps)
    luna = sum(each.luna_cost_usd for each in steps)
    sums = {"minibatch_parent_sum": step.parent_sum, "minibatch_child_sum": step.child_sum}
    return {key: value for key, value in sums.items() if value is not None} | {
        "accepted": float(step.accepted),
        "cost_usd_jev": jev,
        "cost_usd_luna": luna,
        STEP_METRIC: jev + luna,
    }


def _best_metrics(iteration: int, candidates: Sequence[_Candidate]) -> dict[str, float]:
    """Return the scores of the best candidate scored by an iteration, ties to the lowest index."""
    scored = [candidate for candidate in candidates if candidate.iteration <= iteration]
    if not scored:
        return {}
    best = max(scored, key=lambda candidate: (candidate.selection_mean, -candidate.index))
    scores = _scores(best)
    scores["selection_best_score"] = scores.pop("selection_mean")
    return scores


def _live_candidate(line: Mapping[str, object]) -> _Candidate:
    """Read a line of ``candidates.jsonl``; its check is unknown until the export."""
    brier = line["selection_brier"]
    return _Candidate(
        index=int(line["index"]),
        parents=parents_of(line["parents"]),
        iteration=int(line["iteration"]),
        piece=_piece(line["piece"]),
        selection_mean=float(line["selection_mean"]),
        selection_brier={tag: float(brier[tag]) for tag in FOCUS_TAGS},
        words_sha256=str(line["words_sha256"]),
        words=words_from_json(line["words"]),
        passed_check=None,
    )


def _live_step(line: Mapping[str, object]) -> _Step:
    """Read a line of ``history.jsonl``."""
    return _Step(
        iteration=int(line["iteration"]),
        parent_sum=_optional_float(line["parent_sum"]),
        child_sum=_optional_float(line["child_sum"]),
        accepted=bool(line["accepted"]),
        jev_cost_usd=float(line["jev_cost_usd"]),
        luna_cost_usd=float(line["luna_cost_usd"]),
    )


def _record_candidates(record: SearchRecord) -> list[_Candidate]:
    """Read the frozen candidates in index order, with the words ``tuning/`` keeps."""
    return [
        _Candidate(
            index=int(index),
            parents=parents_of(row["parents"]),
            iteration=int(row["iteration"]),
            piece=_piece(row["piece"]),
            selection_mean=float(row["selection_mean"]),
            selection_brier={tag: float(row[f"brier_{metric_suffix(tag)}"]) for tag in FOCUS_TAGS},
            words_sha256=str(row["words_sha256"]),
            words=record.words.get(int(index)),
            passed_check=bool(row["passed_check"]),
        )
        for index, row in record.candidates.sort_index().iterrows()
    ]


def _record_steps(history: pd.DataFrame) -> list[_Step]:
    """Read the frozen history in iteration order."""
    return [
        _Step(
            iteration=int(iteration),
            parent_sum=_optional_float(row["parent_sum"]),
            child_sum=_optional_float(row["child_sum"]),
            accepted=bool(row["accepted"]),
            jev_cost_usd=float(row["jev_cost_usd"]),
            luna_cost_usd=float(row["luna_cost_usd"]),
        )
        for iteration, row in history.sort_index().iterrows()
    ]


def _piece(value: object) -> str | None:
    """Read a changed piece: None for the seed (None, NaN or an empty text)."""
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    """Read a number that may be missing (None or NaN)."""
    return None if pd.isna(value) else float(value)


def _flat_params(settings: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten nested settings into MLflow parameters, ``outer.inner``; non-text values as JSON."""
    flat: dict[str, str] = {}
    for key, value in settings.items():
        if isinstance(value, Mapping):
            flat |= _flat_params(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value if isinstance(value, str) else json.dumps(value)
    return flat


def _check_params(run: Run, params: Mapping[str, str], tracking_uri: str) -> None:
    """Refuse a search run whose parameters are not these settings."""
    logged = dict(run.data.params)
    differ = sorted(
        key for key in logged.keys() | params.keys() if logged.get(key) != params.get(key)
    )
    if differ:
        raise ValueError(
            f"The search run {run.info.run_id} of {tracking_uri} holds other search settings "
            f"({differ}): it is another search of {GEPA_NAME}. Rebuild the store "
            "(notebooks/README.md, Figures) to log this one."
        )


def _all_versions(client: MlflowClient, name: str) -> list[PromptVersion]:
    """Return every version of a prompt, page after page, ``VERSIONS_PAGE_SIZE`` at a time."""
    versions: list[PromptVersion] = []
    token = None
    while True:
        page = client.search_prompt_versions(name, max_results=VERSIONS_PAGE_SIZE, page_token=token)
        versions.extend(page)
        token = page.token
        if not token:
            return versions


def _check_content_tags(name: str, versions: Sequence[PromptVersion], tracking_uri: str) -> None:
    """Refuse a prompt holding a version that no search log wrote: one with no content tag."""
    untagged = sorted(
        int(version.version) for version in versions if CONTENT_TAG not in version.tags
    )
    if untagged:
        raise ValueError(
            f"The prompt {name} of {tracking_uri} holds the versions {untagged} with no "
            f"{CONTENT_TAG} tag: no search log wrote them, so none of their texts can be matched. "
            "Rebuild the store (notebooks/README.md, Figures) to log the search."
        )


def _log_batch(
    client: MlflowClient,
    run_id: str,
    *,
    params: Mapping[str, str] | None = None,
    metrics: Mapping[str, float] | None = None,
    step: int = 0,
) -> None:
    """Log parameters, and metrics at one step, to a run in one call."""
    from mlflow.entities import Metric, Param  # pylint: disable=import-outside-toplevel

    timestamp = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[Metric(key, value, timestamp, step) for key, value in (metrics or {}).items()],
        params=[Param(key, value) for key, value in (params or {}).items()],
    )


def _text_sha256(text: str) -> str:
    """Return the sha256 of a text's UTF-8 bytes: a prompt version's content tag."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
