"""The audit's steps, each reading and writing files in one working folder.

1. ``build_pairs``: the pairs to judge and the batch files the first judge reads. By default
   these are the detector's stage-1 candidates alone, so judging grows with the candidates;
   ``recall_check`` adds the tiers below the threshold that the recorded audit judged once.
2. ``run_pass``, once per pass in ``prompts.PASSES`` and in that order: render the prompts,
   run the judges, and add their answers to ``judgments_raw.json``.
3. ``merge``: the verdicts, ``judgments.csv`` and ``confirmed_pairs.csv``.

The judges' agents run in the ``texts/`` subfolder, which holds only the pair texts, and may
read nothing outside it: the first judge's answers stay out of the second judge's reach.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from codeforces_tags.config import AUDIT_PAIRS_PER_JUDGE
from codeforces_tags.near_duplicates.audit.candidates import (
    candidate_pairs,
    judging_tiers,
    pair_items,
    signal_matrices,
    stage_1_pairs,
    write_batches,
)
from codeforces_tags.near_duplicates.audit.judges import (
    JudgeSettings,
    Query,
    check_labels,
    judge_all,
    sdk_version,
)
from codeforces_tags.near_duplicates.audit.prompts import PASSES, labels_schema, render_prompt
from codeforces_tags.near_duplicates.audit.verdicts import (
    confirmed_pairs,
    labels_frame,
    merge_judgments,
    round_1_pairs,
    round_2_pairs,
)

RAW = "judgments_raw.json"
TEXTS = "texts"  # the only folder a judge may read
# Keys of judgments_raw.json written by each pass: the answers, then the pair ids asked for.
RAW_KEYS = {
    "first_pass": ("first", None),
    "second_pass_round_1": ("second", "sent_to_second"),
    "second_pass_round_2": ("second_round_2", "sent_to_second_round_2"),
}


def build_pairs(df: pd.DataFrame, folder: Path, recall_check: bool = False) -> pd.DataFrame:
    """Write the pairs to judge and the first judge's batch files.

    Args:
        df: The preprocessed dataset, from ``load_dataset()``.
        folder: Working folder of the audit.
        recall_check: Also judge pairs below the candidate threshold, flagged by other
            signals or drawn at random (tiers B to D), to check that the threshold misses no
            duplicate, as the recorded audit did once. Their signals need square matrices,
            so this does not scale; it also writes every flagged pair to ``candidates.csv``.

    Returns:
        The pairs to judge, as saved in ``pairs_to_judge.csv``.
    """
    folder = folder.resolve()
    texts = folder / TEXTS
    texts.mkdir(parents=True, exist_ok=True)
    for old_batch in texts.glob("*.json"):
        old_batch.unlink()
    if recall_check:
        signals = signal_matrices(df)
        candidates = candidate_pairs(df.index.to_numpy(), signals)
        candidates.to_csv(folder / "candidates.csv", index=False)
        pairs = judging_tiers(df, candidates, signals)
    else:
        pairs = stage_1_pairs(df)
    pairs.to_csv(folder / "pairs_to_judge.csv", index=False)
    write_batches(pair_items(df, pairs), texts, "batch", _batch_count(len(pairs)))
    return pairs


def pass_prompts(pass_name: str, folder: Path) -> list[tuple[str, list[int]]]:
    """Render the prompts of a judging pass.

    Round 2 also writes its batch files, ``texts/second_batch_<k>.json``.

    Args:
        pass_name: One of ``prompts.PASSES``.
        folder: Working folder of the audit, after the earlier steps.

    Returns:
        One ``(prompt, pair ids to label)`` per judge.

    Raises:
        ValueError: If an earlier pass has not run yet.
    """
    folder = folder.resolve()
    texts = folder / TEXTS
    batches = sorted(texts.glob("batch_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    if pass_name == "first_pass":
        return [
            (render_prompt(pass_name, batch_file=str(path)), _pair_ids(path)) for path in batches
        ]
    judged = pd.read_csv(folder / "pairs_to_judge.csv", usecols=["pair_id", "word"])
    raw = _require_passes(folder, PASSES[: PASSES.index(pass_name)])
    if pass_name == "second_pass_round_1":
        chosen = round_1_pairs(labels_frame(raw["first"]), judged.set_index("pair_id")["word"])
        prompt = render_prompt(
            pass_name,
            first_batch_file=str(batches[0]),
            last_batch_file=str(batches[-1]),
            pair_ids=chosen,
        )
        return [(prompt, chosen)]
    items = {item["pair_id"]: item for path in batches for item in _read_json(path)}
    chosen = round_2_pairs(judged["pair_id"].tolist(), raw["sent_to_second"])
    paths = write_batches(
        [items[pair_id] for pair_id in chosen], texts, "second_batch", _batch_count(len(chosen))
    )
    return [(render_prompt(pass_name, batch_file=str(path)), _pair_ids(path)) for path in paths]


def run_pass(
    pass_name: str,
    folder: Path,
    dry_run: bool = False,
    settings: JudgeSettings | None = None,
    query: Query | None = None,
) -> list[str]:
    """Run the judges of one pass and save their answers.

    Writes the prompts to ``prompts/<pass>_<k>.txt``, and each judge's answer, cost, token
    usage and tool calls to ``answers/<pass>_<k>.json`` as soon as it is valid. Answers already
    saved for the same prompt are reused, so running the pass again only judges the batches
    that failed. Once every batch has an answer, they go to ``judgments_raw.json``, and the
    agent options, SDK version and total cost to ``run_<pass>.json``.

    Args:
        pass_name: One of ``prompts.PASSES``; passes run in that order.
        folder: Working folder of the audit.
        dry_run: Only write the prompts, without calling any model.
        settings: Model, effort and tools; defaults to the recorded run's.
        query: Sends a prompt; defaults to the Claude Agent SDK.

    Returns:
        The prompts, in judge order.

    Raises:
        RuntimeError: If some judges failed; their batches are listed, the others are saved.
    """
    folder, settings = folder.resolve(), settings or JudgeSettings()
    prompts = pass_prompts(pass_name, folder)
    texts = [prompt for prompt, _ in prompts]
    for k, prompt in enumerate(texts):
        _write_text(folder / "prompts" / f"{pass_name}_{k}.txt", prompt)
    if dry_run:
        return texts
    started = datetime.now(UTC).isoformat(timespec="seconds")
    failures = _judge_missing(pass_name, folder, prompts, settings, query)
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(prompts)} judges of {pass_name} failed; run the pass "
            f"again to judge only them. {failures}"
        )
    answers = [_read_json(_answer_file(folder, pass_name, k)) for k in range(len(prompts))]
    _save_raw(pass_name, folder, [pair_ids for _, pair_ids in prompts], answers)
    _write_json(
        folder / f"run_{pass_name}.json",
        _run_record(pass_name, started, settings.options({}, folder / TEXTS), answers),
    )
    return texts


def merge(folder: Path) -> pd.DataFrame:
    """Merge the judges' answers into verdicts and save them.

    Args:
        folder: Working folder of the audit, after every judging pass.

    Returns:
        The verdicts, as saved in ``judgments.csv`` (``confirmed_pairs.csv`` keeps the
        duplicates).

    Raises:
        ValueError: If a judging pass has not run, or a pair lacks one judge's label.
    """
    folder = folder.resolve()
    raw = _require_passes(folder, PASSES)
    judgments = merge_judgments(pd.read_csv(folder / "pairs_to_judge.csv"), raw)
    unlabelled = judgments[["judge1_label", "judge2_label"]].isna().any(axis=1)
    if unlabelled.any():
        raise ValueError(f"{unlabelled.sum()} pairs lack a judge's label: rerun the passes.")
    judgments.to_csv(folder / "judgments.csv", index=False)
    confirmed_pairs(judgments).to_csv(folder / "confirmed_pairs.csv", index=False)
    return judgments


def read_raw(folder: Path) -> dict[str, list]:
    """Read the judges' answers saved so far.

    Args:
        folder: Working folder of the audit.

    Returns:
        The content of ``judgments_raw.json``, empty before the first pass.
    """
    path = folder / RAW
    return _read_json(path) if path.exists() else {}


def _require_passes(folder: Path, passes: tuple[str, ...]) -> dict[str, list]:
    """Read the saved answers, failing clearly if one of ``passes`` has not run."""
    raw = read_raw(folder)
    for name in passes:
        if RAW_KEYS[name][0] not in raw:
            raise ValueError(f"Run the judging pass {name} first.")
    return raw


def _judge_missing(
    pass_name: str,
    folder: Path,
    prompts: list[tuple[str, list[int]]],
    settings: JudgeSettings,
    query: Query | None,
) -> dict[int, str]:
    """Judge the prompts with no saved answer; save each valid answer, return the failures."""
    todo = [
        k for k, (prompt, _) in enumerate(prompts) if not _has_answer(folder, pass_name, k, prompt)
    ]
    runs = judge_all(
        [prompts[k][0] for k in todo], labels_schema(), settings, folder / TEXTS, query
    )
    failures = {}
    for k, run in zip(todo, runs, strict=True):
        error = run if isinstance(run, Exception) else _labels_error(run.labels, prompts[k][1])
        if error:
            failures[k] = f"{type(error).__name__}: {error}"
            continue
        answer = {"prompt_sha256": _sha256(prompts[k][0])} | asdict(run)
        _write_json(_answer_file(folder, pass_name, k), answer)
    return failures


def _labels_error(labels: list[dict], expected: list[int]) -> ValueError | None:
    """Return why the labels do not match their batch, or None when they do."""
    try:
        check_labels(labels, expected)
    except ValueError as error:
        return error
    return None


def _has_answer(folder: Path, pass_name: str, k: int, prompt: str) -> bool:
    """Tell whether batch ``k`` already has a saved answer to this exact prompt."""
    path = _answer_file(folder, pass_name, k)
    return path.exists() and _read_json(path)["prompt_sha256"] == _sha256(prompt)


def _save_raw(pass_name: str, folder: Path, asked: list[list[int]], answers: list[dict]) -> None:
    """Add a pass's answers to the raw file, dropping the later passes that relied on old ones."""
    position = PASSES.index(pass_name)
    raw = _require_passes(folder, PASSES[:position]) if position else {}
    for later in PASSES[position:]:
        for key in RAW_KEYS[later]:
            raw.pop(key, None)
    answers_key, ids_key = RAW_KEYS[pass_name]
    raw[answers_key] = [label for answer in answers for label in answer["labels"]]
    if ids_key:
        raw[ids_key] = [pair_id for pair_ids in asked for pair_id in pair_ids]
    _write_json(folder / RAW, raw)


def _run_record(pass_name: str, started: str, options: dict, answers: list[dict]) -> dict:
    """Describe a pass's run: agent options (without the schema), SDK version, total cost."""
    del options["output_format"]
    return {
        "pass": pass_name,
        "started_at": started,
        "options": options,
        "claude_agent_sdk": sdk_version(),
        "total_cost_usd": sum(answer["cost_usd"] or 0 for answer in answers),
    }


def _batch_count(n_pairs: int) -> int:
    """Number of batches that gives each judge at most ``AUDIT_PAIRS_PER_JUDGE`` pairs."""
    return max(1, math.ceil(n_pairs / AUDIT_PAIRS_PER_JUDGE))


def _answer_file(folder: Path, pass_name: str, k: int) -> Path:
    """Path of judge ``k``'s saved answer."""
    return folder / "answers" / f"{pass_name}_{k}.json"


def _pair_ids(batch_file: Path) -> list[int]:
    """List the pair ids of a batch file."""
    return [item["pair_id"] for item in _read_json(batch_file)]


def _sha256(text: str) -> str:
    """Fingerprint a prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, content: object) -> None:
    """Write JSON the way the audit's files are written."""
    _write_text(path, json.dumps(content, ensure_ascii=False, indent=1) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Write a text file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
