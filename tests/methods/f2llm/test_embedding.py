"""Tests for the f2llm inputs, the shared embedding cache and the feature frame."""

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.cache import ArrayCache, cache_key
from codeforces_tags.config import ARRAY_CACHE_DIR, FOCUS_TAGS
from codeforces_tags.methods.f2llm.embedding import (
    INPUTS,
    SOLUTION,
    STATEMENT,
    cached_feature_frame,
    cached_vectors,
    embedding_columns,
    encoder_inputs,
    feature_frame,
    fill_cache,
    input_texts,
    live_feature_frame,
    open_cache,
    text_key,
)
from codeforces_tags.methods.f2llm.encoder import (
    EncoderConfig,
    PretrainedRecord,
    frozen_encoder,
    load_tokenizer,
)
from codeforces_tags.methods.f2llm.settings import (
    CACHE_NAME,
    ENCODER_REPO_ID,
    ENCODER_REVISION,
    POOLING,
)
from codeforces_tags.model.features import statement_text
from codeforces_tags.preprocessing.pipeline import raw_to_problems

WIDTH = 4  # of the fake encoder's vectors


def fake_vector(text):
    """Return the fake encoder's float32 vector of a text, drawn from the text's hash."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    return np.random.default_rng(seed).normal(size=WIDTH).astype(np.float32)


def make_fake_encode():
    """Return a stand-in for ``Encoder.encode``, and the list of the batches it was given."""
    batches = []

    def encode(texts):
        batches.append(list(texts))
        return np.stack([fake_vector(text) for text in texts])

    return encode, batches


@pytest.fixture(name="cache")
def cache_fixture(tmp_path, monkeypatch):
    """Return an empty cache under ``tmp_path``, with the read counter at 0."""
    monkeypatch.setattr(ArrayCache, "reads", 0)
    return ArrayCache(tmp_path / "cache")


@pytest.fixture(name="record")
def record_fixture():
    """Return the record of the pinned F2LLM weights (nothing is read from disk)."""
    return PretrainedRecord.from_settings()


@pytest.fixture(name="config")
def config_fixture():
    """Return the encoder configuration of the settings."""
    return EncoderConfig.from_settings()


def test_the_synthetic_problems_give_every_focus_tag_a_positive_in_every_fold(
    synthetic_raw, synthetic_labels
):
    positives = synthetic_labels.groupby(synthetic_raw["fold"]).sum()

    assert positives.index.tolist() == [0, 1, 2, 3, 4]
    assert (positives >= 1).all().all()
    assert list(synthetic_labels.columns) == list(FOCUS_TAGS)
    assert int((synthetic_labels.sum(axis=1) == 0).sum()) == 5  # one no-focus problem per fold
    assert synthetic_raw.groupby("group")["fold"].nunique().max() == 1  # groups stay in a fold


def test_the_synthetic_statements_read_differently_in_the_tiny_vocabulary(
    synthetic_raw, tiny_pretrained, tiny_config
):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)
    texts = encoder_inputs(raw_to_problems(synthetic_raw), tiny_config)[STATEMENT]

    whole = tokenizer(texts)["input_ids"]
    # As Encoder.encode gives them to the model: cut to max_length tokens, the eos last.
    cut = tokenizer(texts, truncation=True, max_length=tiny_config.max_length)["input_ids"]

    assert all(tokenizer.unk_token_id not in row for row in whole)  # every word is known
    assert len({tuple(row) for row in cut}) == len(synthetic_raw)  # no two read the same, once cut


def test_input_texts_are_the_statement_with_its_notes_and_the_source_code(raw_problems):
    problems = raw_to_problems(raw_problems)

    texts = input_texts(problems)

    assert list(texts) == list(INPUTS) == [STATEMENT, SOLUTION]
    assert texts[STATEMENT] == statement_text(problems, include_notes=True)
    assert texts[STATEMENT][0] == problems["statement"].iloc[0]  # problem 30 has no notes
    assert texts[STATEMENT][1].endswith(problems["prob_desc_notes"].iloc[1])
    assert texts[SOLUTION] == problems["source_code"].tolist()


def test_encoder_inputs_prepend_each_inputs_own_instruction(raw_problems, config):
    problems = raw_to_problems(raw_problems)
    config = dataclasses.replace(config, statement_instruction="S: ", solution_instruction="C: ")

    inputs = encoder_inputs(problems, config)
    texts = input_texts(problems)

    assert inputs[STATEMENT] == ["S: " + text for text in texts[STATEMENT]]
    assert inputs[SOLUTION] == ["C: " + text for text in texts[SOLUTION]]


def test_embedding_columns_name_each_dimension_of_an_input():
    assert embedding_columns(STATEMENT, 3) == [
        "statement_emb_0000",
        "statement_emb_0001",
        "statement_emb_0002",
    ]
    assert embedding_columns(SOLUTION, 1) == ["solution_emb_0000"]
    assert embedding_columns(SOLUTION, 0) == []


def test_feature_frame_appends_float64_embedding_columns_to_the_problems(synthetic_raw):
    raw = synthetic_raw.iloc[:3]
    vectors = {
        STATEMENT: np.arange(6, dtype=np.float32).reshape(3, 2),
        SOLUTION: np.ones((3, 3), dtype=np.float32),
    }

    frame = feature_frame(raw, vectors)

    problems = raw_to_problems(raw)
    statement, solution = embedding_columns(STATEMENT, 2), embedding_columns(SOLUTION, 3)
    assert list(frame.columns) == [*problems.columns, *statement, *solution]
    assert frame.index.equals(raw.index)  # out of sample id order
    assert (frame[[*statement, *solution]].dtypes == "float64").all()
    np.testing.assert_array_equal(frame[statement], vectors[STATEMENT])
    np.testing.assert_array_equal(frame[solution], vectors[SOLUTION])
    pd.testing.assert_frame_equal(frame[problems.columns], problems)
    assert frame[["fold", "group"]].equals(raw[["fold", "group"]])


def test_feature_frame_takes_some_of_the_inputs_but_refuses_an_unknown_one(synthetic_raw):
    raw = synthetic_raw.iloc[:2]

    frame = feature_frame(raw, {STATEMENT: np.zeros((2, 2))})

    assert list(frame.columns[-2:]) == embedding_columns(STATEMENT, 2)
    assert not any(column.startswith("solution_emb") for column in frame.columns)
    with pytest.raises(ValueError, match=r"Unknown inputs \['notes'\]"):
        feature_frame(raw, {STATEMENT: np.zeros((2, 2)), "notes": np.zeros((2, 2))})


def test_feature_frame_refuses_vectors_of_another_number_of_problems(synthetic_raw):
    with pytest.raises(ValueError, match="Shape of passed values"):
        feature_frame(synthetic_raw.iloc[:3], {STATEMENT: np.zeros((2, 2))})


def test_text_key_hashes_the_weights_the_pooling_the_length_and_the_text(record, config):
    key = text_key("an encoder input", record, config)

    assert key == cache_key(ENCODER_REPO_ID, ENCODER_REVISION, POOLING, "1024", "an encoder input")
    assert key != text_key("an encoder input", record, dataclasses.replace(config, max_length=512))
    other_revision = dataclasses.replace(record, revision="1" * 40)
    assert key != text_key("an encoder input", other_revision, config)
    assert key != text_key("another encoder input", record, config)


def test_open_cache_is_the_f2llm_folder_of_the_array_cache():
    assert open_cache().directory == ARRAY_CACHE_DIR / CACHE_NAME == ARRAY_CACHE_DIR / "f2llm"


def test_fill_cache_embeds_each_distinct_text_once_chunk_by_chunk(cache, record, config):
    encode, batches = make_fake_encode()
    lines = []

    computed = fill_cache(
        ["a", "b", "a", "c", "d", "e"], cache, encode, record, config, chunk=2, log=lines.append
    )

    assert computed == 5
    assert batches == [["a", "b"], ["c", "d"], ["e"]]
    assert len(list(cache.directory.iterdir())) == 5
    assert [line.split(";")[0] for line in lines] == [
        "2/5 distinct encoder inputs cached",
        "4/5 distinct encoder inputs cached",
        "5/5 distinct encoder inputs cached",
    ]
    assert "2 embedded" in lines[0]
    assert "min left" in lines[0]


def test_a_second_fill_computes_nothing(cache, record, config):
    encode, batches = make_fake_encode()
    fill_cache(["a", "b", "c"], cache, encode, record, config, chunk=2, log=[].append)
    lines = []

    computed = fill_cache(["c", "b", "a"], cache, encode, record, config, chunk=2, log=lines.append)

    assert computed == 0
    assert batches == [["a", "b"], ["c"]]  # only the first fill's
    assert "0 embedded" in lines[-1]
    assert "left" not in lines[-1]


def test_an_interrupted_fill_resumes_after_its_last_written_chunk(cache, record, config):
    encode, batches = make_fake_encode()
    calls = []

    def interrupted(texts):
        calls.append(list(texts))
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return encode(texts)

    texts = ["a", "b", "c", "d", "e"]
    with pytest.raises(RuntimeError, match="interrupted"):
        fill_cache(texts, cache, interrupted, record, config, chunk=2, log=[].append)

    computed = fill_cache(texts, cache, encode, record, config, chunk=2, log=[].append)

    assert computed == 3
    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_fill_cache_of_no_text_computes_and_logs_nothing(cache, record, config):
    encode, batches = make_fake_encode()
    lines = []

    assert fill_cache([], cache, encode, record, config, log=lines.append) == 0
    assert (batches, lines) == ([], [])
    assert not cache.directory.exists()


def test_cached_vectors_reads_float32_rows_in_the_order_of_the_texts(cache, record, config):
    encode, _ = make_fake_encode()
    fill_cache(["a", "b"], cache, encode, record, config, log=[].append)

    vectors = cached_vectors(["b", "a", "b"], cache, record, config)

    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, np.stack([fake_vector(text) for text in "bab"]))
    assert ArrayCache.reads == 3


def test_cached_vectors_never_encodes_a_missing_text(cache, record, config):
    encode, _ = make_fake_encode()
    fill_cache(["a"], cache, encode, record, config, log=[].append)
    shorter = dataclasses.replace(config, max_length=512)

    with pytest.raises(
        LookupError, match="1 encoder inputs .* codeforces_tags.methods.f2llm embed"
    ):
        cached_vectors(["a", "z"], cache, record, config)
    with pytest.raises(LookupError, match="no cached embedding"):
        cached_vectors(["a"], cache, record, shorter)  # another max length is another key
    assert len(list(cache.directory.iterdir())) == 1


def test_cached_vectors_of_no_text_is_refused(cache, record, config):
    with pytest.raises(ValueError, match="no keys"):
        cached_vectors([], cache, record, config)


def test_the_cached_and_the_live_feature_frames_agree(
    synthetic_raw, tiny_pretrained, tiny_config, cache
):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    raw = synthetic_raw.iloc[:6]
    inputs = encoder_inputs(raw_to_problems(raw), tiny_config)

    computed = fill_cache(
        [*inputs[STATEMENT], *inputs[SOLUTION]],
        cache,
        encoder.encode,
        tiny_pretrained,
        tiny_config,
        log=[].append,
    )
    cached = cached_feature_frame(raw, cache, tiny_pretrained, tiny_config)
    live = live_feature_frame(raw, encoder, encoder)

    columns = [
        *embedding_columns(STATEMENT, encoder.width),
        *embedding_columns(SOLUTION, encoder.width),
    ]
    assert computed == 12
    assert list(cached.columns) == list(live.columns)
    assert list(cached.columns[-len(columns) :]) == columns
    np.testing.assert_allclose(cached[columns], live[columns], rtol=0, atol=1e-5)
    pd.testing.assert_frame_equal(cached.drop(columns=columns), live.drop(columns=columns))
    np.testing.assert_allclose(np.linalg.norm(live[columns[: encoder.width]], axis=1), 1, atol=1e-6)


def test_a_live_feature_frame_without_a_solution_encoder_has_no_solution_columns(
    synthetic_raw, tiny_pretrained, tiny_config
):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    frame = live_feature_frame(synthetic_raw.iloc[:2], encoder, None)

    assert list(frame.columns[-encoder.width :]) == embedding_columns(STATEMENT, encoder.width)
    assert not any(column.startswith("solution_emb") for column in frame.columns)
