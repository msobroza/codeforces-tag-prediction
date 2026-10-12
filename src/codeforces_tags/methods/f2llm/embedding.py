"""Embed problems with the frozen encoder: the inputs, the shared cache and the feature frame.

Each problem has two inputs, embedded separately: the statement input, which is the text the
frozen TF-IDF statement block reads (``statement_text(problems, include_notes=True)``), and the
solution input, the source code. An encoder input is an input with its instruction prepended:
the exact string the tokenizer receives.

The cache is ``ArrayCache(ARRAY_CACHE_DIR / CACHE_NAME)``, one folder shared by the four f2llm
methods rather than the documented ``ARRAY_CACHE_DIR / <method name>``: a key hashes the
weights (repository and revision), the pooling, the maximum length and the exact encoder input,
so a cached vector is the same whichever method reads it, and one ``embed`` pass serves all
four. ``fill_cache`` is the only function that writes it. Fit-time reads go through
``cached_vectors``, whose callback raises: a missing embedding never turns into silent
encoding. A loaded method has no cache: it encodes live (``live_feature_frame``).

The feature frame is what every estimator of this package fits on and scores: the preprocessed
problems, then one float64 column per dimension of each input's embedding.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn

import numpy as np
import pandas as pd

from codeforces_tags.cache import ArrayCache, cache_key
from codeforces_tags.config import ARRAY_CACHE_DIR
from codeforces_tags.methods.f2llm.encoder import Encoder, EncoderConfig, PretrainedRecord
from codeforces_tags.methods.f2llm.settings import CACHE_NAME, CHUNK_TEXTS
from codeforces_tags.model.features import statement_text
from codeforces_tags.preprocessing.pipeline import raw_to_problems

STATEMENT = "statement"
SOLUTION = "solution"
INPUTS: tuple[str, ...] = (STATEMENT, SOLUTION)
_EMBED_COMMAND = "python -m codeforces_tags.methods.f2llm embed"


def input_texts(problems: pd.DataFrame) -> dict[str, list[str]]:
    """Return the two inputs of each problem, before their instructions.

    Args:
        problems: Preprocessed problems (``raw_to_problems``), with ``statement``,
            ``prob_desc_notes`` and ``source_code``.

    Returns:
        ``{"statement": ..., "solution": ...}``, one text per problem in row order: the
        statement with its notes after a blank line, when it has any, and the source code.
    """
    return {
        STATEMENT: statement_text(problems, include_notes=True),
        SOLUTION: problems["source_code"].tolist(),
    }


def encoder_inputs(problems: pd.DataFrame, config: EncoderConfig) -> dict[str, list[str]]:
    """Return the encoder inputs of each problem: each input with its instruction prepended.

    Args:
        problems: Preprocessed problems (``raw_to_problems``).
        config: The encoder configuration, which holds the two instructions.

    Returns:
        ``{"statement": ..., "solution": ...}``, one string per problem in row order.
    """
    instructions = {STATEMENT: config.statement_instruction, SOLUTION: config.solution_instruction}
    return {
        name: [instructions[name] + text for text in texts]
        for name, texts in input_texts(problems).items()
    }


def embedding_columns(input_name: str, width: int) -> list[str]:
    """Name the columns of one input's embedding in the feature frame.

    Args:
        input_name: ``"statement"`` or ``"solution"``.
        width: The embedding's dimensions.

    Returns:
        ``["<input>_emb_0000", "<input>_emb_0001", ...]``, ``width`` names.
    """
    return [f"{input_name}_emb_{dimension:04d}" for dimension in range(width)]


def feature_frame(raw_problems: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build the feature frame of raw problems from their embeddings.

    Args:
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``; extra columns (the
            ``fold`` and ``group`` that ``select_part`` appends, ``tags``) pass through.
        vectors: One array per input, among ``INPUTS``: one row per problem, in row order.

    Returns:
        ``raw_to_problems(raw_problems)``, then each given input's embedding as float64 columns
        (``embedding_columns``, in ``INPUTS`` order), the width taken from its array.

    Raises:
        ValueError: If ``vectors`` names an input outside ``INPUTS``, or if an array has another
            number of rows than there are problems.
    """
    return _with_embeddings(raw_to_problems(raw_problems), vectors)


def text_key(text: str, record: PretrainedRecord, config: EncoderConfig) -> str:
    """Return the cache key of one encoder input.

    Args:
        text: The encoder input, its instruction included.
        record: The pretrained weights, whose repository and revision the key holds.
        config: The encoder configuration, whose pooling and maximum length the key holds.

    Returns:
        ``cache_key(repo_id, revision, pooling, str(max_length), text)``.
    """
    return cache_key(record.repo_id, record.revision, config.pooling, str(config.max_length), text)


def open_cache() -> ArrayCache:
    """Open the cache of the frozen encoder's embeddings, shared by the four f2llm methods.

    Returns:
        ``ArrayCache(ARRAY_CACHE_DIR / CACHE_NAME)``; its folder is created at its first write.
    """
    return ArrayCache(ARRAY_CACHE_DIR / CACHE_NAME)


def fill_cache(  # pylint: disable=too-many-arguments
    texts: Sequence[str],
    cache: ArrayCache,
    encode: Callable[[list[str]], np.ndarray],
    record: PretrainedRecord,
    config: EncoderConfig,
    *,
    chunk: int = CHUNK_TEXTS,
    log: Callable[[str], None] = print,
) -> int:
    """Embed the encoder inputs the cache lacks, chunk by chunk.

    The distinct texts go to ``get_or_compute_many`` ``chunk`` at a time; it hands ``encode``
    the missing ones of the chunk only, and writes their vectors once ``encode`` returns. A
    second run computes nothing, and an interrupted run loses at most its last chunk.

    Args:
        texts: Encoder inputs; a text may repeat (it is embedded once).
        cache: The cache, e.g. ``open_cache()``.
        encode: Embeds a list of encoder inputs, one row each, e.g. ``Encoder.encode``.
        record: The pretrained weights of ``encode``'s encoder (part of each key).
        config: The encoder configuration of ``encode``'s encoder (part of each key).
        chunk: Distinct texts per call to ``get_or_compute_many``.
        log: Called with one progress line per chunk.

    Returns:
        How many texts were embedded: 0 when every one was cached.
    """
    by_key = {text_key(text, record, config): text for text in texts}
    keys = list(by_key)
    embedded = 0

    def compute(missing: list[str]) -> np.ndarray:
        nonlocal embedded
        vectors = encode([by_key[key] for key in missing])
        embedded += len(missing)
        return vectors

    started = time.perf_counter()
    for begin in range(0, len(keys), chunk):
        cache.get_or_compute_many(keys[begin : begin + chunk], compute)
        done = min(begin + chunk, len(keys))
        log(_progress(done, len(keys), embedded, time.perf_counter() - started))
    return embedded


def cached_vectors(
    texts: Sequence[str], cache: ArrayCache, record: PretrainedRecord, config: EncoderConfig
) -> np.ndarray:
    """Read the cached embeddings of encoder inputs, never encoding one.

    Args:
        texts: Encoder inputs, at least one; a text may repeat.
        cache: The cache that ``fill_cache`` filled.
        record: The pretrained weights the vectors were computed with.
        config: The encoder configuration the vectors were computed with.

    Returns:
        Float32 ``(len(texts), width)``, in the order of ``texts``.

    Raises:
        LookupError: If a text has no cached vector; the message names the ``embed`` command.
        ValueError: If ``texts`` is empty.
    """
    keys = [text_key(text, record, config) for text in texts]
    return cache.get_or_compute_many(keys, _refuse_to_encode).astype(np.float32, copy=False)


def cached_feature_frame(
    raw_problems: pd.DataFrame, cache: ArrayCache, record: PretrainedRecord, config: EncoderConfig
) -> pd.DataFrame:
    """Build the feature frame of raw problems from the cache: the fit-time path.

    Args:
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``, at least one.
        cache: The cache that ``fill_cache`` filled.
        record: The pretrained weights the vectors were computed with.
        config: The encoder configuration the vectors were computed with.

    Returns:
        The ``feature_frame`` of both inputs.

    Raises:
        LookupError: If an encoder input has no cached vector.
    """
    problems = raw_to_problems(raw_problems)
    inputs = encoder_inputs(problems, config)
    vectors = {name: cached_vectors(inputs[name], cache, record, config) for name in INPUTS}
    return _with_embeddings(problems, vectors)


def live_feature_frame(
    raw_problems: pd.DataFrame, statement_encoder: Encoder, solution_encoder: Encoder | None
) -> pd.DataFrame:
    """Build the feature frame of raw problems by encoding them now: the prediction path.

    Args:
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.
        statement_encoder: Encodes the statement inputs (SetFit's merged body, or the frozen
            encoder), with the instruction of its own configuration.
        solution_encoder: Encodes the solution inputs; None leaves them out, for a method
            whose estimator gives the solution embedding no weight.

    Returns:
        The ``feature_frame`` of the encoded inputs.
    """
    problems = raw_to_problems(raw_problems)
    encoders = {STATEMENT: statement_encoder, SOLUTION: solution_encoder}
    vectors = {
        name: encoder.encode(encoder_inputs(problems, encoder.config)[name])
        for name, encoder in encoders.items()
        if encoder is not None
    }
    return _with_embeddings(problems, vectors)


def _with_embeddings(problems: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Append each given input's embedding to preprocessed problems, as float64 columns."""
    unknown = sorted(set(vectors) - set(INPUTS))
    if unknown:
        raise ValueError(f"Unknown inputs {unknown}: the inputs are {list(INPUTS)}.")
    blocks = [
        pd.DataFrame(
            np.asarray(vectors[name], dtype=np.float64),
            index=problems.index,
            columns=embedding_columns(name, np.shape(vectors[name])[1]),
        )
        for name in INPUTS
        if name in vectors
    ]
    return pd.concat([problems, *blocks], axis=1)


def _refuse_to_encode(missing: list[str]) -> NoReturn:
    """Raise: fit-time code reads the cache, and never encodes what it lacks."""
    raise LookupError(
        f"{len(missing)} encoder inputs have no cached embedding: run {_EMBED_COMMAND} first, "
        "which embeds every problem once."
    )


def _progress(done: int, total: int, embedded: int, seconds: float) -> str:
    """Describe how far ``fill_cache`` got, and at most how long it still needs."""
    line = f"{done:,}/{total:,} distinct encoder inputs cached; {embedded:,} embedded"
    line += f" in {seconds:.0f} s"
    if not embedded:
        return line
    rate = embedded / max(seconds, 1e-9)
    return f"{line} ({rate:.1f} per s, at most {(total - done) / rate / 60:.1f} min left)"
