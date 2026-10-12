"""Command line of the F2LLM methods: weights, embeddings, tuning, score_atol, freeze, verify.

    python -m codeforces_tags.methods.f2llm fetch [--method NAME]
    python -m codeforces_tags.methods.f2llm embed [--device cuda]
    python -m codeforces_tags.methods.f2llm reference
    python -m codeforces_tags.methods.f2llm tune --method NAME
    python -m codeforces_tags.methods.f2llm export --method NAME
    python -m codeforces_tags.methods.f2llm atol [--device cuda]
    python -m codeforces_tags.methods.f2llm freeze --method NAME [--device cuda] [--allow-dirty]
    python -m codeforces_tags.methods.f2llm verify --method NAME... [--retrain] [--single]
        [--device cpu]

Every command but fetch takes --data DIR and --split FILE; freeze and verify also take
--assets DIR (the frozen methods), and freeze --tracking-uri URI and --artifact-dir DIR (the
MLflow store). Run them with the f2llm group, and with HF_HUB_OFFLINE=1 but for fetch. embed,
atol and freeze (cuda by default) and verify --retrain on cuda are batched GPU work, the load
that rebooted a machine once: run them only when that work is meant.

- fetch downloads the pinned encoder files and checks their sha256. The four methods share them,
  so --method, which the command line's fetch hint names, changes nothing.
- embed embeds both encoder inputs of every problem once into the shared array cache, then
  writes input_lengths.csv and the embedding pairs to assets/f2llm/ once (later runs check).
- reference writes TF-IDF's reference rows to assets/f2llm/ once, if they reproduce its frozen
  dev and test macro F1.
- tune runs a tuned method's grid study on the cached dev embeddings (resumable); export writes
  the configuration it chose to experiments/<name>/tuning/.
- atol measures how far each method's scores move from cached (batched) to live (one problem
  at a time, CPU) embeddings, refit without LATENCY_PROBLEMS drawn dev problems; SetFit's
  adapter comes from a short smoke training. It prints the score_atol to register.
- freeze scores dev out of fold from the cache (a tuned method's must reproduce the macro F1 its
  export recorded), then test once, saves and checks the method folder (on the CPU), copies
  tuning/, writes predictions.csv, scores.csv and run_record.json once, and logs to MLflow. A
  second run re-derives nothing (score_atol > 0): it checks, then logs.
- verify, writing nothing, re-derives the dev rows and thresholds from scores.csv and
  re-predicts the test rows live with the method folder, on the CPU unless --device names cuda
  or auto (cuda when torch sees a GPU, as a loaded method); --single scores each test problem in
  its own call, as the command line predicts one exercise, and the live re-prediction on the
  GPU always does (never the sustained full power of one batch of every test problem, which
  rebooted a machine once);
  --retrain also refits from the cache.
  Scores agree within score_atol; a tag flips only within score_atol of its threshold.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from codeforces_tags.config import (
    FOCUS_TAGS,
    LATENCY_PROBLEMS,
    LATENCY_SEED,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
    TUNING_BLAS_THREADS,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.conformance import check_frozen_method, check_method_folder
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    load_method,
    save_method,
    sibling_methods,
)
from codeforces_tags.methods.f2llm.analysis import (
    INPUT_LENGTHS_FILE,
    PAIR_COUNTS_FILE,
    PAIRS_FILE,
    TFIDF_REFERENCE_FILE,
    embedding_pairs,
    input_lengths,
    tfidf_reference,
    write_or_check,
)
from codeforces_tags.methods.f2llm.drift import score_gaps, setfit_score_gap, suggest_score_atol
from codeforces_tags.methods.f2llm.embedding import (
    SOLUTION,
    STATEMENT,
    cached_feature_frame,
    cached_vectors,
    encoder_inputs,
    fill_cache,
    live_feature_frame,
    open_cache,
)
from codeforces_tags.methods.f2llm.encoder import (
    Encoder,
    EncoderConfig,
    PretrainedRecord,
    fetch,
    frozen_encoder,
    resolve_device,
    token_counts,
)
from codeforces_tags.methods.f2llm.estimators import make_estimator, tfidf_params
from codeforces_tags.methods.f2llm.method import from_fit
from codeforces_tags.methods.f2llm.setfit import CUBLAS_WORKSPACE_CONFIG, SetFitEstimator
from codeforces_tags.methods.f2llm.settings import (
    AUTO_DEVICE,
    CPU_THREADS,
    EARLY_FUSION,
    GPU_BATCH_TOKENS,
    GRIDS,
    LATE_FUSION,
    NAMES,
    PROBE,
    SCORE_ATOL_FACTOR,
    SETFIT,
    SETFIT_METHOD,
    SETFIT_SMOKE_PAIRS,
    SETTINGS,
    SHARED_ASSETS_DIR,
    TFIDF_TEST_MACRO_F1,
    TUNED,
)
from codeforces_tags.methods.f2llm.tuning import (
    TUNING_FOLDER,
    check_exported_score,
    export,
    journal_path,
    load_params,
    open_study,
    run_study,
    tuning_dir,
)
from codeforces_tags.methods.parts import Parts, add_dataset_arguments, read_parts
from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    apply_thresholds,
    cross_fitted_predictions,
    per_tag_thresholds,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.provenance import (
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    make_run_record,
    sha256_json,
)
from codeforces_tags.splitting import load_split
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.methods import log_method, part_metrics
from codeforces_tags.tuning.results import load_tuning_results

# The devices the encoder runs on.
_DEVICES: tuple[str, ...] = ("cpu", "cuda")
# The methods whose recipe and estimator use the frozen TF-IDF configuration, and those that
# inherit the probe's chosen configuration (late fusion fuses the probe; SetFit's head is it).
_USES_TFIDF = (LATE_FUSION, EARLY_FUSION)
_INHERITS_PROBE = (LATE_FUSION, SETFIT_METHOD)
# The dev problems freeze scores with the saved method folder, live, to check it against the
# refit: a few, since each is encoded on the CPU, where score_atol was measured.
_FOLDER_CHECK_PROBLEMS = 3


class _AtolRows(NamedTuple):
    """The rows atol fits on and scores."""

    fit: pd.DataFrame  # cached feature frame of the dev problems not drawn
    labels: np.ndarray  # their focus labels, int64
    reference: pd.DataFrame  # cached feature frame of the drawn problems
    live: pd.DataFrame  # feature frame of the drawn problems, each encoded alone on CPU
    raw: pd.DataFrame  # the drawn raw problems, with the columns a method reads


class _Frames(NamedTuple):
    """The feature frames of the dev and test problems, read from the cache."""

    dev: pd.DataFrame
    test: pd.DataFrame


class _Rows(NamedTuple):
    """One part's rows of a score method: its float scores and its 0/1 predictions."""

    scores: pd.DataFrame
    predictions: pd.DataFrame


class _Frozen(NamedTuple):
    """A frozen method's rows, and the thresholds verify re-derives from its dev scores."""

    dev: _Rows
    test: _Rows
    fold_thresholds: pd.DataFrame  # one row per dev fold, chosen on the other folds
    thresholds: pd.Series  # chosen on every dev score: the thresholds of the test rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``; for every command but fetch ``data`` and ``split``;
        ``method`` for fetch (None by default, and unused: the four methods share one encoder),
        tune, export, freeze and verify (a list for verify); ``device`` for embed, atol, freeze
        and verify; ``assets`` for freeze and verify; ``allow_dirty``, ``tracking_uri`` and
        ``artifact_dir`` for freeze; ``retrain`` and ``single`` for verify.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.methods.f2llm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="download the pinned encoder files and check them")
    download.add_argument(
        "--method",
        choices=NAMES,
        help="any F2LLM method, as the command line names it: the four share one encoder, so "
        "every name fetches the same files",
    )
    embed = commands.add_parser("embed", help="embed every problem once into the cache")
    commands.add_parser("reference", help="write the frozen TF-IDF reference rows once")
    for name, help_text in (
        ("tune", "run a tuned method's grid study"),
        ("export", "write the configuration a study chose"),
    ):
        tuned = commands.add_parser(name, help=help_text)
        tuned.add_argument("--method", required=True, choices=TUNED, help="a tuned method")
    atol = commands.add_parser("atol", help="measure the score gap of cached and live encoding")
    freeze = commands.add_parser("freeze", help="freeze the files once, then log to MLflow")
    freeze.add_argument("--method", required=True, choices=NAMES, help="the method to freeze")
    freeze.add_argument(
        "--allow-dirty", action="store_true", help="freeze from uncommitted code (recorded)"
    )
    add_store_arguments(freeze)
    verify = commands.add_parser("verify", help="re-derive the frozen files of methods")
    verify.add_argument("--method", required=True, nargs="+", choices=NAMES, help="methods")
    verify.add_argument("--retrain", action="store_true", help="also refit from the cache")
    verify.add_argument(
        "--single",
        action="store_true",
        help="score each test problem in its own call (always so on the GPU)",
    )
    for command in (embed, atol, freeze):
        command.add_argument("--device", choices=_DEVICES, default="cuda", help="where to run")
    verify.add_argument(  # the CPU unless asked: auto is where a loaded method encodes
        "--device",
        choices=(*_DEVICES, AUTO_DEVICE),
        default="cpu",
        help="where to run (auto: cuda when torch sees a GPU, else cpu)",
    )
    for command in (freeze, verify):
        add_dataset_arguments(command)  # --data, --split and --assets, as every local method's
    for name, command in commands.choices.items():
        if name not in ("fetch", "freeze", "verify"):
            command.add_argument("--data", help="dataset folder (default: $CODEFORCES_DATA_DIR)")
            command.add_argument("--split", type=Path, default=SPLIT_FILE, help="frozen split")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of the F2LLM methods.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status, 0; a refused dataset, a dirty tree, a missing embedding or a frozen
        file that the code no longer reproduces raises.
    """
    # Before any CUDA work: cuBLAS reads it when its first handle is made, and a SetFit fit
    # that trains after other GPU work in this process (verify --retrain) is deterministic only
    # if it was set by then. A str literal key, as the MLflow-variable test of the package asks.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    args = parse_args(argv)
    commands = {
        "fetch": _fetch,
        "embed": _embed,
        "reference": _reference,
        "tune": _tune,
        "export": _export,
        "atol": _atol,
        "freeze": _freeze,
        "verify": _verify,
    }
    return commands[args.command](args)


def recipe_sha256(name: str, params: dict) -> str:
    """Hash what defines a method: its name, tags, settings, configuration and weights.

    Never a fitted value, which is an outcome of the data (docs/adr/0004).

    Args:
        name: The method's registry key.
        params: Its configuration, as freeze reads it: the chosen grid point (its
            ``tuning/params.json``), and, for late fusion and SetFit, the probe's under
            ``"probe"``.

    Returns:
        The sha256 hex digest of the name, the focus tags, ``SETTINGS``, ``params``, the
        pinned weights' ``pretrained.json`` and, for the two fusion methods, the frozen TF-IDF
        configuration.
    """
    recipe = {
        "method": name,
        "tags": list(FOCUS_TAGS),
        "settings": SETTINGS,
        "params": params,
        "pretrained": _encoding()[0].to_dict(),
    }
    if name in _USES_TFIDF:
        recipe["tfidf"] = tfidf_params().to_dict()
    return sha256_json(recipe)


def supersedes(assets: Path, name: str) -> list[str]:
    """Name the sibling methods whose predictions are committed, as the freeze records them.

    The siblings are the other F2LLM methods (``sibling_methods``, ADR 0004); methods frozen
    before any of them is committed give ``[]``.

    Args:
        assets: The folder of the frozen methods.
        name: The method being frozen.

    Returns:
        The sorted names of the siblings whose ``predictions.csv`` is tracked by git and
        unchanged since ``HEAD``.
    """
    return sorted(
        other
        for other in sibling_methods(name, METHODS)
        if committed_unchanged(assets / other / PREDICTIONS_FILE)
    )


def _encoding() -> tuple[PretrainedRecord, EncoderConfig]:
    """Return what the commands encode with: the pinned weights and the settings' configuration.

    The tests replace it with the tiny encoder's.
    """
    return PretrainedRecord.from_settings(), EncoderConfig.from_settings()


def _dev_frame(parts: Parts) -> pd.DataFrame:
    """Return the dev problems' feature frame, from the cache: it never encodes."""
    return cached_feature_frame(parts.dev, open_cache(), *_encoding())


def _labels(parts: Parts, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the focus labels of a feature frame's problems, in its order."""
    return parts.labels.loc[frame.index]


def _width(frame: pd.DataFrame) -> int:
    """Return the embedding width of a feature frame: its statement embedding columns."""
    return sum(column.startswith(f"{STATEMENT}_emb_") for column in frame.columns)


def _params(name: str, own: Path, probe: Path) -> dict:
    """Read a method's configuration from tuning/ folders: its own, and the probe's if it needs."""
    params = dict(load_params(name, own)) if name in TUNED else {}
    if name in _INHERITS_PROBE:
        params["probe"] = load_params(PROBE, probe)
    return params


def _study_inputs(name: str, frame: pd.DataFrame) -> dict:
    """Return what run_study and export take besides the rows: width, probe and TF-IDF."""
    return {
        "width": _width(frame),
        "probe_params": load_params(PROBE, tuning_dir(PROBE)) if name == LATE_FUSION else None,
        "tfidf": tfidf_params() if name in _USES_TFIDF else None,
    }


def _estimator(name: str, params: dict, *, width: int, device: str):
    """Build a method's unfitted score estimator from its configuration."""
    if name == SETFIT_METHOD:
        return _setfit_estimator(params, device, settings=SETFIT, resume=True)
    own = {key: value for key, value in params.items() if key != "probe"}
    tfidf = tfidf_params() if name in _USES_TFIDF else None
    return make_estimator(name, own, width=width, probe_params=params.get("probe"), tfidf=tfidf)


def _setfit_estimator(params: dict, device: str, *, settings: dict, resume: bool):
    """Build SetFit's unfitted estimator: the probe's head, trained on ``device``."""
    record, config = _encoding()
    return SetFitEstimator(
        head_params=params["probe"],
        settings=settings,
        device=device,
        pretrained=record.to_dict(),
        encoder_config=config.to_dict(),
        resume=resume,
    )


def _print_gpu_memory(device: str) -> None:
    """Print the peak GPU memory torch allocated, when the work ran on the GPU."""
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


@contextmanager
def _device_environment(device: str) -> Iterator[None]:
    """Make the methods loaded inside encode on ``device``; restore the variable after."""
    # A str literal key, config.DEVICE_ENV_VAR's value: tests/tracking/test_environment.py
    # counts an os.environ write under any other key as a possible MLflow variable, which only
    # mlflow_env may set.
    previous = os.environ.get("CODEFORCES_DEVICE")
    os.environ["CODEFORCES_DEVICE"] = device
    try:
        yield
    finally:
        if previous is None:
            del os.environ["CODEFORCES_DEVICE"]
        else:
            os.environ["CODEFORCES_DEVICE"] = previous


def _fetch(args: argparse.Namespace) -> int:  # pylint: disable=unused-argument
    """Download the pinned encoder files, then check them."""
    record, _ = _encoding()
    directory = fetch(record)
    print(
        f"the {len(record.files)} files of {record.repo_id} at {record.revision} are in "
        f"{directory}, each with its pinned sha256"
    )
    return 0


def _embed(args: argparse.Namespace) -> int:
    """Embed every problem once into the cache, then write the shared files once."""
    check_dataset(args.data)
    raw = read_samples(args.data)
    record, config = _encoding()
    device = resolve_device(args.device)
    encoder = frozen_encoder(record, config, device)
    texts = encoder_inputs(raw_to_problems(raw), config)
    cache = open_cache()
    inputs = texts[STATEMENT] + texts[SOLUTION]
    computed = fill_cache(inputs, cache, _counting_tokens(encoder), record, config)
    print(f"embedded {computed:,} encoder inputs on {device}; the others were cached")
    _print_gpu_memory(device)
    SHARED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    lengths = input_lengths(raw, encoder.tokenizer, config)
    print(write_or_check(lengths, SHARED_ASSETS_DIR / INPUT_LENGTHS_FILE))
    vectors = cached_vectors(texts[STATEMENT], cache, record, config)
    pairs, counts = embedding_pairs(pd.DataFrame(vectors, index=raw.index), load_split(args.split))
    print(write_or_check(pairs, SHARED_ASSETS_DIR / PAIRS_FILE))
    print(write_or_check(counts, SHARED_ASSETS_DIR / PAIR_COUNTS_FILE))
    return 0


def _counting_tokens(encoder: Encoder) -> Callable[[list[str]], np.ndarray]:
    """Wrap ``encoder.encode`` to print the tokens it encodes per second, chunk by chunk."""

    def encode(texts: list[str]) -> np.ndarray:
        started = time.perf_counter()
        vectors = encoder.encode(texts)
        seconds = max(time.perf_counter() - started, 1e-9)
        counts = np.minimum(token_counts(texts, encoder.tokenizer), encoder.config.max_length)
        tokens = int(counts.sum())
        print(f"encoded {tokens:,} tokens in {seconds:.1f} s: {tokens / seconds:,.0f} per s")
        return vectors

    return encode


def _reference(args: argparse.Namespace) -> int:
    """Write the frozen TF-IDF reference rows once, if they reproduce its frozen F1."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    frame = tfidf_reference(
        parts.dev,
        parts.test,
        parts.labels,
        tfidf_params(),
        expected_dev_macro_f1=float(load_tuning_results().out_of_fold.at["macro_f1", "f1"]),
        expected_test_macro_f1=TFIDF_TEST_MACRO_F1,
    )
    SHARED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(write_or_check(frame, SHARED_ASSETS_DIR / TFIDF_REFERENCE_FILE))
    return 0


def _tune(args: argparse.Namespace) -> int:
    """Run a tuned method's grid study on the cached dev embeddings; it resumes."""
    check_dataset(args.data)
    name = args.method
    parts = read_parts(args.data, args.split)
    frame = _dev_frame(parts)
    journal = journal_path(name)
    study = run_study(
        name,
        frame,
        _labels(parts, frame),
        frame["fold"],
        **_study_inputs(name, frame),
        journal=journal,
    )
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"{name}: {len(complete)} of {len(GRIDS[name])} configurations scored, in {journal}")
    return 0


def _export(args: argparse.Namespace) -> int:
    """Write the configuration a tuned method's study chose, with its trials."""
    check_dataset(args.data)
    name = args.method
    journal = journal_path(name)
    if not journal.is_file():
        raise FileNotFoundError(f"{journal} does not exist: run `tune --method {name}` first.")
    parts = read_parts(args.data, args.split)
    frame = _dev_frame(parts)
    directory = export(
        name,
        open_study(name, journal),
        frame,
        _labels(parts, frame),
        frame["fold"],
        **_study_inputs(name, frame),
        directory=tuning_dir(name),
    )
    print(f"exported the configuration {name}'s study chose to {directory}")
    return 0


def _atol(args: argparse.Namespace) -> int:
    """Measure how far each method's scores move from cached to live embeddings."""
    check_dataset(args.data)
    rows = _atol_rows(read_parts(args.data, args.split))
    gaps = {name: _tuned_gap(name, rows) for name in TUNED}
    gaps[SETFIT_METHOD] = _setfit_gap(rows, resolve_device(args.device))
    for name, gap in gaps.items():
        print(f"{name}: the scores move by up to {gap:.3g} from cached to live embeddings")
    print(
        f"score_atol {suggest_score_atol(gaps.values()):g}: {SCORE_ATOL_FACTOR} times the largest "
        "gap, rounded up; write it into _F2LLM_SCORE_ATOL in codeforces_tags/methods/__init__.py"
    )
    return 0


def _atol_rows(parts: Parts) -> _AtolRows:
    """Draw the problems atol scores, and build their cached and live feature frames."""
    record, config = _encoding()
    drawn = parts.dev.sample(n=LATENCY_PROBLEMS, random_state=LATENCY_SEED).sort_index()
    cache = open_cache()
    fit = cached_feature_frame(parts.dev.drop(index=drawn.index), cache, record, config)
    encoder = frozen_encoder(record, config, "cpu")
    problems = drawn[list(RAW_INPUT_COLUMNS)]
    live = [live_feature_frame(problems.iloc[[row]], encoder, encoder) for row in range(len(drawn))]
    return _AtolRows(
        fit=fit,
        labels=_labels(parts, fit).to_numpy(dtype=np.int64),
        reference=cached_feature_frame(drawn, cache, record, config),
        live=pd.concat(live),
        raw=problems,
    )


def _tuned_gap(name: str, rows: _AtolRows) -> float:
    """Measure a tuned method's largest score gap, its configuration read from experiments/."""
    params = _params(name, tuning_dir(name), tuning_dir(PROBE))
    width = _width(rows.fit)
    return score_gaps(
        lambda: _estimator(name, params, width=width, device="cpu"),
        rows.fit,
        rows.labels,
        rows.reference,
        rows.live,
    )


def _setfit_gap(rows: _AtolRows, device: str) -> float:
    """Measure SetFit's largest score gap, with an adapter from a short smoke training.

    The adapter is trained and merged on ``device`` and scores the cached rows in batches; the
    live side is a loaded method, merged on CPU, scoring each raw problem alone.
    """
    params = _params(SETFIT_METHOD, tuning_dir(SETFIT_METHOD), tuning_dir(PROBE))
    settings = _smoke_settings()
    with _device_environment("cpu"):  # where the live side merges its adapter
        gap = setfit_score_gap(
            lambda: _setfit_estimator(params, device, settings=settings, resume=False),
            rows.fit,
            rows.labels,
            rows.reference,
            rows.raw,
        )
    _print_gpu_memory(device)
    return gap


def _smoke_settings() -> dict:
    """Return SetFit's settings with about ``SETFIT_SMOKE_PAIRS`` pairs in all.

    A positive and a negative pair are drawn together, for each focus tag.
    """
    per_tag = max(1, SETFIT_SMOKE_PAIRS // (2 * len(FOCUS_TAGS)))
    return SETFIT | {"pairs_per_tag_positive": per_tag}


def _cached_frames(parts: Parts) -> _Frames:
    """Return the dev and test feature frames, from the cache."""
    (record, config), cache = _encoding(), open_cache()
    return _Frames(
        dev=cached_feature_frame(parts.dev, cache, record, config),
        test=cached_feature_frame(parts.test, cache, record, config),
    )


def _fit(
    name: str, params: dict, frame: pd.DataFrame, labels: pd.DataFrame, device: str
) -> CrossFittedTagThresholds:
    """Score dev out of fold with cross-fitted thresholds, then refit on all of dev."""
    estimator = _estimator(name, params, width=_width(frame), device=device)
    with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
        return CrossFittedTagThresholds(estimator, refit=True).fit(
            frame, labels, groups=frame["fold"]
        )


def _dev_rows(fitted: CrossFittedTagThresholds, frame: pd.DataFrame) -> _Rows:
    """Return the dev rows of a fit: its out-of-fold scores and cross-fitted predictions."""
    tags = list(FOCUS_TAGS)
    scores = pd.DataFrame(fitted.oof_scores_, index=frame.index, columns=tags)
    predictions = pd.DataFrame(fitted.oof_predictions_, index=frame.index, columns=tags)
    return _Rows(scores.astype(np.float64), predictions)


def _test_rows(fitted: CrossFittedTagThresholds, frame: pd.DataFrame) -> _Rows:
    """Return the test rows of a fit: the refit's scores, and its predictions at thresholds_."""
    tags = list(FOCUS_TAGS)
    scores = pd.DataFrame(
        np.asarray(fitted.decision_function(frame), dtype=np.float64),
        index=frame.index,
        columns=tags,
    )
    return _Rows(scores, apply_thresholds(scores, pd.Series(fitted.thresholds_, index=tags)))


def _freeze(args: argparse.Namespace) -> int:
    """Freeze a method's files once, or check them; then log it to MLflow."""
    name = args.method
    if METHODS[name].score_atol <= 0:
        raise ValueError(
            f"{name} is registered with score_atol {METHODS[name].score_atol}: the f2llm methods "
            "are not bit-reproducible, so freeze needs the score_atol that atol measured."
        )
    check_dataset(args.data)
    check_clean_tree(args.allow_dirty)
    directory = args.assets / name
    _refuse_half_frozen(directory)
    parts = read_parts(args.data, args.split)
    if (directory / PREDICTIONS_FILE).exists():
        _check_frozen(name, args.assets, parts)
    else:
        _derive_and_freeze(name, args, parts)
    # What is frozen, not a record made now: a second run's has another created_at.
    record = load_run_record(directory)
    metrics = part_metrics(parts.labels, load_predictions(directory), FOCUS_TAGS)
    store = {"tracking_uri": args.tracking_uri, "artifact_dir": args.artifact_dir}
    model_id = log_method(
        directory / MODEL_FOLDER, record, metrics, split=parts.split, registry=METHODS, **store
    )
    print(f"logged {name} to MLflow: model {model_id}")
    return 0


def _refuse_half_frozen(directory: Path) -> None:
    """Refuse frozen predictions without the method folder that freeze saves before them."""
    folder = directory / MODEL_FOLDER
    if (directory / PREDICTIONS_FILE).exists() and not folder.exists():
        files = [
            name
            for name in (PREDICTIONS_FILE, SCORES_FILE, RUN_RECORD_FILE)
            if (directory / name).exists()
        ]
        raise ValueError(
            f"{directory} holds {files} but no method folder {MODEL_FOLDER}/, which freeze saves "
            f"first. If they are committed, restore it from git (git checkout -- {folder}); if "
            f"they were never committed, delete {', '.join(files)} and freeze again."
        )


def _check_frozen(name: str, assets: Path, parts: Parts) -> None:
    """Check a frozen method's files and recipe, re-deriving nothing (score_atol > 0)."""
    check_frozen_method(name, assets_dir=assets, split=parts.split, registry=METHODS)
    _check_recipe(name, assets)
    print(
        f"{assets / name} is frozen: its files are complete and hold the recipe of the code "
        f"(score_atol {METHODS[name].score_atol:g}, so nothing is re-derived)"
    )


def _check_recipe(name: str, assets: Path) -> dict:
    """Fail unless the frozen run record holds the recipe of the code; return the params."""
    params = _params(name, assets / name / TUNING_FOLDER, assets / PROBE / TUNING_FOLDER)
    stored = load_run_record(assets / name).recipe_sha256
    expected = recipe_sha256(name, params)
    if stored != expected:
        raise ValueError(
            f"{assets / name} was frozen with another recipe ({stored}, not {expected}): the "
            f"settings, the chosen configuration or the weights of {name} changed since."
        )
    return params


def _derive_and_freeze(name: str, args: argparse.Namespace, parts: Parts) -> None:
    """Fit, check it against its export, score test once, save the folder, then freeze once."""
    device = resolve_device(args.device)
    params = _params(name, tuning_dir(name), args.assets / PROBE / TUNING_FOLDER)
    frames = _cached_frames(parts)
    labels = _labels(parts, frames.dev)
    fitted = _fit(name, params, frames.dev, labels, device)
    dev = _dev_rows(fitted, frames.dev)
    if name in TUNED:  # before the test set is scored and any file written
        print(check_exported_score(name, labels, dev.predictions, tuning_dir(name)))
    test = _test_rows(fitted, frames.test)
    directory = args.assets / name
    print(_freeze_folder(name, fitted, directory / MODEL_FOLDER, frames.dev, parts.dev))
    print(_copy_tuning(name, directory))
    record = make_run_record(
        name,
        recipe_sha256(name, params),
        allow_dirty=args.allow_dirty,
        requirements=METHODS[name].requirements,
        method_entries=_method_entries(name, params, device, args.assets),
        data_dir=args.data,
        split_path=args.split,
    )
    print(
        freeze_predictions(
            stack_parts(dev=dev.predictions, test=test.predictions),
            directory,
            record,
            scores=stack_parts(dev=dev.scores, test=test.scores),
            score_atol=METHODS[name].score_atol,
        )
    )


def _freeze_folder(
    name: str,
    fitted: CrossFittedTagThresholds,
    folder: Path,
    frame: pd.DataFrame,
    dev: pd.DataFrame,
) -> str:
    """Save the method folder once; then check it, and its live scores against the refit's.

    Both build the method on the CPU, whatever torch sees: the gap to the refit is held to
    score_atol, which was measured on the CPU.
    """
    saved = not (folder / METHOD_FILE).exists()
    with _device_environment("cpu"):
        if saved:
            record, config = _encoding()
            save_method(from_fit(name, fitted, config, record), folder)
        check_method_folder(folder)
        gap = _folder_gap(name, fitted, folder, frame, dev)
    action = "saved the method folder" if saved else "the method folder"
    return (
        f"{action} {folder}: it scores {_FOLDER_CHECK_PROBLEMS} dev problems live within "
        f"{gap:.3g} of the refit"
    )


def _folder_gap(
    name: str,
    fitted: CrossFittedTagThresholds,
    folder: Path,
    frame: pd.DataFrame,
    dev: pd.DataFrame,
) -> float:
    """Score a few dev problems with the loaded folder, live, and with the refit, cached."""
    spec = METHODS[name]
    rows = dev.index[:_FOLDER_CHECK_PROBLEMS]
    loaded = load_method(folder, registry=METHODS).scores(dev.loc[rows, list(spec.input_columns)])
    refit = np.asarray(fitted.decision_function(frame.loc[rows]), dtype=np.float64)
    gap = float(np.abs(loaded.to_numpy(dtype=np.float64) - refit).max())
    atol = max(spec.score_atol, TUNING_EXPORT_TOLERANCE)
    if not gap <= atol:  # not >: a NaN gap is never within atol
        raise ValueError(
            f"The method folder {folder} scores {len(rows)} dev problems up to {gap:.3g} away "
            f"from the refit, beyond {atol:g} (score_atol): the folder comes from another fit, "
            "or its scores are not numbers. Delete it if it was never committed, then freeze "
            "again."
        )
    return gap


def _copy_tuning(name: str, directory: Path) -> str:
    """Copy a tuned method's tuning/ folder next to its method folder once; then check it."""
    if name not in TUNED:
        return f"{name} tunes nothing on dev: it has no {TUNING_FOLDER}/ folder"
    source, target = tuning_dir(name), directory / TUNING_FOLDER
    if not target.exists():
        shutil.copytree(source, target)
        return f"copied {source} to {target}"
    if _file_bytes(source) != _file_bytes(target):
        raise ValueError(
            f"{target} differs from {source}: the study was exported again since the first "
            "freeze. Freeze the new configuration as a new method, or restore the copy."
        )
    return f"{target} holds the results of {source}"


def _file_bytes(folder: Path) -> dict[str, bytes]:
    """Map each file under a folder to its bytes."""
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _method_entries(name: str, params: dict, device: str, assets: Path) -> dict:
    """Return what this package adds to a run record (CLAUDE.md, Tagging methods, item 6)."""
    batch_tokens = GPU_BATCH_TOKENS if device == "cuda" else _encoding()[1].cpu_batch_tokens
    return {
        "settings": SETTINGS,
        "params": params,
        "dev_in_sample": False,
        "supersedes": supersedes(assets, name),
        "device": device,
        "dtype": "float32",
        "batch_tokens": batch_tokens,
        "threads": CPU_THREADS,
    }


def _verify(args: argparse.Namespace) -> int:
    """Check each method's frozen files against the dataset, the code and its folder."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    for name in args.method:
        _verify_method(name, args, parts)
    return 0


def _verify_method(name: str, args: argparse.Namespace, parts: Parts) -> None:
    """Run verify's level (a), and (b) under --retrain, for one method; print its scores."""
    directory = args.assets / name
    check_frozen_method(name, assets_dir=args.assets, split=parts.split, registry=METHODS)
    params = _check_recipe(name, args.assets)
    print(f"{name}: the frozen files are complete and hold the recipe of the code")
    frozen = _check_offline(name, directory, parts)
    print(_check_live(name, args, frozen, parts))
    if args.retrain:
        _check_retrain(name, args, params, frozen, parts)
    scores = part_metrics(parts.labels, load_predictions(directory), FOCUS_TAGS)
    for dataset, metrics in scores.items():
        print(
            f"{name} {dataset}: macro F1 {metrics['macro_f1']:.4f}, "
            f"micro F1 {metrics['micro_f1']:.4f}"
        )


def _check_offline(name: str, directory: Path, parts: Parts) -> _Frozen:
    """Re-derive the dev rows and the thresholds from scores.csv, and the test rows from them."""
    scores, predictions = load_scores(directory), load_predictions(directory)
    dev = _Rows(scores.loc["dev"], predictions.loc["dev"])
    test = _Rows(scores.loc["test"], predictions.loc["test"])
    labels = parts.labels.loc[dev.scores.index]
    folds = parts.dev["fold"].loc[dev.scores.index]
    fold_thresholds, dev_predictions = cross_fitted_predictions(dev.scores, labels, folds)
    if not dev_predictions.equals(dev.predictions):
        raise ValueError(
            f"The dev rows of {name}'s {PREDICTIONS_FILE} are not the ones its {SCORES_FILE} "
            "gives with cross-fitted thresholds."
        )
    thresholds = per_tag_thresholds(dev.scores, labels)
    if not apply_thresholds(test.scores, thresholds).equals(test.predictions):
        raise ValueError(
            f"The test rows of {name}'s {PREDICTIONS_FILE} are not its test scores under the "
            f"thresholds its dev scores give."
        )
    print(
        f"{name}: the dev rows and the thresholds re-derive from {SCORES_FILE}, and the test "
        "rows apply them"
    )
    return _Frozen(dev, test, fold_thresholds, thresholds)


def _check_live(name: str, args: argparse.Namespace, frozen: _Frozen, parts: Parts) -> str:
    """Re-predict the test rows with the method folder, live, and compare with the frozen."""
    spec = METHODS[name]
    folder = args.assets / name / MODEL_FOLDER
    problems = parts.test.loc[frozen.test.scores.index, list(spec.input_columns)]
    device = resolve_device(args.device)
    single = args.single or device == "cuda"  # never one batch of every test problem on the GPU
    with _device_environment(device):
        method = load_method(folder, registry=METHODS)
        scores = _single_scores(method, problems) if single else method.scores(problems)
    _print_gpu_memory(device)
    if not np.array_equal(method.thresholds, frozen.thresholds.to_numpy(dtype=np.float64)):
        raise ValueError(
            f"The method folder {folder} holds other thresholds than the frozen dev scores give: "
            "the folder and the frozen files come from different fits."
        )
    live = _Rows(scores, apply_thresholds(scores, frozen.thresholds))
    thresholds = _broadcast(frozen.thresholds, scores)
    what = f"{name} test, re-predicted live by {folder}"
    if single:
        what += ", one problem per call"
    return _compare(what, frozen.test, live, thresholds, spec.score_atol)


def _single_scores(method, problems: pd.DataFrame) -> pd.DataFrame:
    """Score each problem in its own call, as the command line predicts one exercise."""
    return pd.concat([method.scores(problems.iloc[[row]]) for row in range(len(problems))])


def _check_retrain(
    name: str, args: argparse.Namespace, params: dict, frozen: _Frozen, parts: Parts
) -> None:
    """Refit from the cache and compare both parts' rows with the frozen ones."""
    frames = _cached_frames(parts)
    device = resolve_device(args.device)
    fitted = _fit(name, params, frames.dev, _labels(parts, frames.dev), device)
    dev, test = _dev_rows(fitted, frames.dev), _test_rows(fitted, frames.test)
    atol = METHODS[name].score_atol
    fold_thresholds = frozen.fold_thresholds.loc[frames.dev["fold"]].to_numpy(dtype=np.float64)
    print(_compare(f"{name} dev, refit from the cache", frozen.dev, dev, fold_thresholds, atol))
    thresholds = _broadcast(frozen.thresholds, test.scores)
    print(_compare(f"{name} test, refit from the cache", frozen.test, test, thresholds, atol))


def _compare(
    what: str, frozen: _Rows, derived: _Rows, thresholds: np.ndarray, score_atol: float
) -> str:
    """Fail unless the derived scores are within score_atol, and flip only near a threshold.

    ``thresholds`` holds the frozen threshold of each derived row and tag; returns a line to
    print, with the largest gap and the number of flips.
    """
    index = derived.scores.index
    expected = frozen.scores.loc[index].to_numpy(dtype=np.float64)
    gaps = np.abs(derived.scores.to_numpy(dtype=np.float64) - expected)
    largest = float(gaps.max(initial=0.0))  # NaN when a gap is NaN
    atol = max(score_atol, TUNING_EXPORT_TOLERANCE)
    if not largest <= atol:  # not >: a NaN gap is never within atol
        raise ValueError(
            f"{what}: the scores differ from the frozen {SCORES_FILE} by up to {largest:.3g} "
            f"({np.count_nonzero(~np.isfinite(gaps))} of them NaN or infinite), beyond "
            f"{atol:g} (score_atol, rtol 0)."
        )
    flips = derived.predictions.to_numpy() != frozen.predictions.loc[index].to_numpy()
    far = np.abs(expected - thresholds) > score_atol
    if (flips & far).any():
        raise ValueError(
            f"{what}: {int((flips & far).sum())} tags flip where the frozen score lies beyond "
            f"score_atol ({score_atol:g}) of its threshold."
        )
    return (
        f"{what}: the scores are within {atol:g} of the frozen ones (largest gap {largest:.3g}); "
        f"{int(flips.sum())} tags flip, each within score_atol of its threshold"
    )


def _broadcast(thresholds: pd.Series, scores: pd.DataFrame) -> np.ndarray:
    """Repeat the per-tag thresholds on every row of a score frame."""
    return np.broadcast_to(thresholds.to_numpy(dtype=np.float64), scores.shape)


if __name__ == "__main__":
    sys.exit(main())
