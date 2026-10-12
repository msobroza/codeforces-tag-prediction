"""Tests for the F2LLM encoder: weights, configuration, devices, loading and encoding."""

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from codeforces_tags.config import DEVICE_ENV_VAR, LEGACY_DEVICE_ENV_VAR, PRETRAINED_DIR_ENV_VAR
from codeforces_tags.methods.f2llm import encoder as encoder_module
from codeforces_tags.methods.f2llm.encoder import (
    Encoder,
    EncoderConfig,
    PretrainedRecord,
    check_no_pickle_files,
    check_pretrained,
    clear_memo,
    encode,
    fetch,
    frozen_encoder,
    load_model,
    load_tokenizer,
    loaded_device,
    read_json,
    repository_files,
    resolve_device,
    token_counts,
    write_json,
)
from codeforces_tags.methods.f2llm.settings import (
    AUTO_DEVICE,
    CPU_BATCH_TOKENS,
    CPU_THREADS,
    ENCODER_FILE_PATTERNS,
    ENCODER_FILES,
    ENCODER_REPO_ID,
    ENCODER_REVISION,
    EOS_TOKEN_ID,
    GPU_BATCH_TOKENS,
    GPU_MAX_BATCH_SEQUENCES,
    MAX_LENGTH,
    POOLING,
    SOLUTION_INSTRUCTION,
    STATEMENT_INSTRUCTION,
)

# A repository as the fake download writes it: two repository files, and the metadata
# snapshot_download keeps under .cache/ (timestamps, which change at every download).
REPOSITORY = {
    "config.json": b'{"hidden_size": 32}',
    "sub/model.safetensors": b"weights",
    ".cache/huggingface/download/config.json.metadata": b"1759795200.0",
}
PINNED = {
    path: hashlib.sha256(data).hexdigest()
    for path, data in REPOSITORY.items()
    if not path.startswith(".cache/")
}


def write_files(directory, files):
    """Write ``{relative path: bytes}`` under ``directory``."""
    for path, data in files.items():
        target = Path(directory) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def fake_download(files, calls):
    """Return a stand-in for ``snapshot_download`` that writes ``files`` and records its call."""

    def download(**kwargs):
        calls.append(kwargs)
        write_files(kwargs["local_dir"], files)

    return download


@pytest.fixture(name="weights_root")
def weights_root_fixture(tmp_path, monkeypatch):
    """Point ``$CODEFORCES_PRETRAINED_DIR`` at a folder under ``tmp_path``, and return it."""
    root = tmp_path / "weights"
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(root))
    return root


@pytest.fixture(name="record")
def record_fixture(weights_root):  # pylint: disable=unused-argument
    """Return the record of ``REPOSITORY``'s pinned files, under the test's weights root."""
    return PretrainedRecord("org/model", "1" * 40, dict(PINNED))


def test_the_encoder_config_of_the_fit_path_holds_the_settings():
    config = EncoderConfig.from_settings()

    assert config.to_dict() == {
        "statement_instruction": STATEMENT_INSTRUCTION,
        "solution_instruction": SOLUTION_INSTRUCTION,
        "max_length": MAX_LENGTH,
        "pooling": POOLING,
        "eos_token_id": EOS_TOKEN_ID,
        "cpu_batch_tokens": CPU_BATCH_TOKENS,
    }


def test_the_encoder_config_round_trips_through_a_sorted_json_file(tmp_path):
    config = EncoderConfig.from_settings()

    write_json(tmp_path / "encoder.json", config.to_dict())

    text = (tmp_path / "encoder.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert list(json.loads(text)) == sorted(config.to_dict())
    assert EncoderConfig.from_dict(read_json(tmp_path / "encoder.json")) == config


def test_the_encoder_config_refuses_missing_or_unexpected_keys():
    values = EncoderConfig.from_settings().to_dict()

    with pytest.raises(ValueError, match=r"missing \['max_length'\], unexpected \[\]"):
        EncoderConfig.from_dict(
            {key: value for key, value in values.items() if key != "max_length"}
        )
    with pytest.raises(ValueError, match=r"missing \[\], unexpected \['width'\]"):
        EncoderConfig.from_dict(values | {"width": 1024})


def test_the_encoder_config_refuses_a_pooling_the_code_does_not_implement():
    values = EncoderConfig.from_settings().to_dict()

    with pytest.raises(ValueError, match="'mean' is not implemented"):
        EncoderConfig.from_dict(values | {"pooling": "mean"})


def test_the_pretrained_record_of_the_settings_pins_the_repository():
    record = PretrainedRecord.from_settings()

    assert (record.repo_id, record.revision, record.files) == (
        ENCODER_REPO_ID,
        ENCODER_REVISION,
        ENCODER_FILES,
    )
    assert record.files is not ENCODER_FILES  # a copy: the settings stay as pinned


def test_the_pretrained_record_lists_its_files_sorted_by_path_and_round_trips(tmp_path):
    record = PretrainedRecord("org/model", "1" * 40, {"b.json": "2" * 64, "a.json": "1" * 64})

    write_json(tmp_path / "pretrained.json", record.to_dict())

    stored = read_json(tmp_path / "pretrained.json")
    assert stored["files"] == [
        {"path": "a.json", "sha256": "1" * 64},
        {"path": "b.json", "sha256": "2" * 64},
    ]
    assert PretrainedRecord.from_dict(stored) == record


def test_the_pretrained_record_refuses_bad_keys_and_repeated_paths():
    values = {"repo_id": "org/model", "revision": "1" * 40, "files": []}

    with pytest.raises(ValueError, match=r"unexpected \['weights'\]"):
        PretrainedRecord.from_dict(values | {"weights": "model.safetensors"})
    with pytest.raises(ValueError, match=r"a file of pretrained.json .* missing \['sha256'\]"):
        PretrainedRecord.from_dict(values | {"files": [{"path": "a.json"}]})
    twice = [{"path": "a.json", "sha256": "1" * 64}, {"path": "a.json", "sha256": "2" * 64}]
    with pytest.raises(ValueError, match="lists a file twice"):
        PretrainedRecord.from_dict(values | {"files": twice})


@pytest.mark.parametrize(
    ("repo_id", "path"), [("../org/model", "a.json"), ("org/model", "/etc/a.json"), ("", "a")]
)
def test_the_pretrained_record_refuses_names_that_leave_the_weights_folder(repo_id, path):
    with pytest.raises(ValueError, match="is not a relative path inside the folder"):
        PretrainedRecord(repo_id, "1" * 40, {path: "1" * 64})


def test_the_weights_folder_is_the_snapshot_folder_of_the_pretrained_cache(weights_root):
    record = PretrainedRecord("org/model", "1" * 40, {})

    assert record.directory() == weights_root / "org--model" / ("1" * 40)


def test_the_weights_folder_refuses_a_revision_that_is_not_one_folder_name(weights_root):
    record = PretrainedRecord("org/model", "main/sub", {})

    with pytest.raises(ValueError, match="The revision 'main/sub'"):
        record.directory()
    assert not weights_root.exists()


def test_the_repository_files_are_hashed_without_the_download_cache(tmp_path):
    write_files(tmp_path, REPOSITORY)

    assert repository_files(tmp_path) == PINNED
    assert list(repository_files(tmp_path)) == ["config.json", "sub/model.safetensors"]


@pytest.mark.parametrize(
    "path", ["pytorch_model.bin", "sub/optimizer.PT", ".cache/huggingface/state.pkl"]
)
def test_a_file_of_a_pickle_format_is_refused_anywhere_in_the_folder(tmp_path, path):
    write_files(tmp_path, REPOSITORY | {path: b"\x80\x04"})

    with pytest.raises(ValueError, match="pickle formats") as error:
        check_no_pickle_files(tmp_path)
    assert path in str(error.value)


def test_check_pretrained_returns_the_folder_of_the_pinned_files(record):
    write_files(record.directory(), REPOSITORY)

    assert check_pretrained(record) == record.directory()


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (REPOSITORY | {"config.json": b"{}"}, r"changed \['config.json'\]"),
        ({"config.json": REPOSITORY["config.json"]}, r"missing \['sub/model.safetensors'\]"),
        (REPOSITORY | {"pytorch_model.bin": b"\x80\x04"}, "pickle formats"),
    ],
)
def test_check_pretrained_refuses_a_changed_missing_or_pickle_file(record, files, message):
    write_files(record.directory(), files)

    with pytest.raises(ValueError, match=message):
        check_pretrained(record)


def test_fetch_downloads_the_pinned_patterns_into_the_record_folder(record):
    calls = []

    directory = fetch(record, download=fake_download(REPOSITORY, calls))

    assert directory == record.directory()
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "1" * 40,
            "allow_patterns": list(ENCODER_FILE_PATTERNS),
            "local_dir": record.directory(),
        }
    ]


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"config.json": b"{}"}, r"changed \['config.json'\]"),
        ({"tokenizer.json": b"{}"}, r"not pinned \['tokenizer.json'\]"),
        ({".cache/huggingface/download/model.pt": b"\x80\x04"}, "pickle formats"),
    ],
)
def test_fetch_refuses_files_that_are_not_the_pinned_ones(record, files, message):
    with pytest.raises(ValueError, match=message):
        fetch(record, download=fake_download(REPOSITORY | files, []))


def test_fetch_refuses_a_download_that_lacks_a_pinned_file(record):
    with pytest.raises(ValueError, match=r"missing \['sub/model.safetensors'\]"):
        fetch(record, download=fake_download({"config.json": REPOSITORY["config.json"]}, []))


def test_fetch_downloads_the_settings_repository_by_default(weights_root):
    calls = []

    with pytest.raises(ValueError, match="does not hold the pinned encoder files"):
        fetch(download=fake_download(REPOSITORY, calls))

    assert (calls[0]["repo_id"], calls[0]["revision"]) == (ENCODER_REPO_ID, ENCODER_REVISION)
    assert calls[0]["local_dir"] == weights_root / "codefuse-ai--F2LLM-0.6B" / ENCODER_REVISION


def test_fetch_downloads_with_snapshot_download_by_default(record, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download(REPOSITORY, calls))

    assert fetch(record) == record.directory()
    assert len(calls) == 1


def test_write_json_writes_sorted_keys_indented_with_a_trailing_newline(tmp_path):
    write_json(tmp_path / "values.json", {"b": [1, 2], "a": "x"})

    text = (tmp_path / "values.json").read_text(encoding="utf-8")
    assert text == '{\n  "a": "x",\n  "b": [\n    1,\n    2\n  ]\n}\n'


def test_the_cpu_is_always_a_device_and_cuda_needs_a_gpu(monkeypatch):
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch sees no GPU"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    with pytest.raises(ValueError, match="'tpu' is not a device"):
        resolve_device("tpu")


@pytest.mark.parametrize(("gpu", "expected"), [(True, "cuda"), (False, "cpu")])
def test_the_automatic_device_is_the_gpu_when_torch_sees_one(monkeypatch, gpu, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)

    assert resolve_device(AUTO_DEVICE) == expected


@pytest.mark.parametrize(("gpu", "expected"), [(True, "cuda"), (False, "cpu")])
@pytest.mark.parametrize("variable", [None, "", AUTO_DEVICE])
def test_a_loaded_method_encodes_on_the_gpu_when_torch_sees_one(
    monkeypatch, variable, gpu, expected
):
    if variable is None:
        monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    else:  # empty, which counts as unset, or the default named explicitly
        monkeypatch.setenv(DEVICE_ENV_VAR, variable)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)

    assert loaded_device() == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_environment_overrides_the_automatic_device(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv(DEVICE_ENV_VAR, device)

    assert loaded_device() == device


def test_the_environment_names_a_device_and_cuda_needs_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")
    with pytest.raises(RuntimeError, match="torch sees no GPU"):
        loaded_device()
    monkeypatch.setenv(DEVICE_ENV_VAR, "gpu")
    with pytest.raises(ValueError, match="'gpu', not a device setting"):
        loaded_device()


def test_the_variable_the_f2llm_methods_read_before_is_refused_naming_the_new_one(monkeypatch):
    # no_gpu set the new variable to "cpu": the former one is refused all the same.
    monkeypatch.setenv(LEGACY_DEVICE_ENV_VAR, "cpu")
    refusal = (
        f"${LEGACY_DEVICE_ENV_VAR} is no longer read: unset it, and name the device in "
        f"${DEVICE_ENV_VAR}"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}"):
        loaded_device()


# Encoder inputs of the tiny vocabulary, with their token counts (the end-of-sequence token
# included): 4, 7 and 2.
TEXTS = ["given a tree", "print the sum of the array", "a"]
# 30 words: 31 tokens, truncated to the tiny config's 16.
LONG_TEXT = " ".join(["count the edge of a tree"] * 5)


def words(count):
    """Return an encoder input of ``count`` known words: ``count + 1`` tokens."""
    return " ".join(["a"] * count)


def alone(model, tokenizer, text, max_length):
    """Embed one text without padding, by hand: the normalised state of its last token."""
    tokens = tokenizer([text], truncation=True, max_length=max_length, return_tensors="pt")
    assert tokens["input_ids"][0, -1] == tokenizer.eos_token_id
    with torch.inference_mode():
        state = model(**tokens).last_hidden_state[0, -1]
    return torch.nn.functional.normalize(state, dim=0).numpy()


def batch_shapes(model):
    """Record the shape of every batch ``model`` is called on; return the list it fills."""
    shapes = []

    def record(_module, _args, kwargs):
        shapes.append(tuple(kwargs["input_ids"].shape))

    model.register_forward_pre_hook(record, with_kwargs=True)
    return shapes


def test_the_tiny_tokenizer_appends_its_end_token_and_pads_on_the_right(
    tiny_pretrained, tiny_config
):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)

    tokens = tokenizer(["given a tree", "a"], padding=True)["input_ids"]

    end, pad = tiny_config.eos_token_id, tokenizer.pad_token_id
    assert [row[-1] for row in tokens] == [end, pad]
    assert tokens[1][1:] == [end, pad, pad]


def test_truncation_keeps_the_end_token_last(tiny_pretrained, tiny_config):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)

    tokens = tokenizer([LONG_TEXT], truncation=True, max_length=tiny_config.max_length)

    assert len(tokens["input_ids"][0]) == tiny_config.max_length
    assert tokens["input_ids"][0][-1] == tiny_config.eos_token_id


def test_load_tokenizer_refuses_another_end_token_or_left_padding(
    tiny_pretrained, tiny_config, tmp_path
):
    directory = tiny_pretrained.directory()
    with pytest.raises(ValueError, match="end-of-sequence token is 1, but encoder.json says 2"):
        load_tokenizer(directory, dataclasses.replace(tiny_config, eos_token_id=2))

    left = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    left.padding_side = "left"
    left.save_pretrained(tmp_path / "left")

    with pytest.raises(ValueError, match="pads on the left"):
        load_tokenizer(tmp_path / "left", tiny_config)


def test_load_model_upcasts_the_stored_bfloat16_weights_for_inference(tiny_pretrained):
    directory = tiny_pretrained.directory()

    model = load_model(directory, "cpu")

    assert read_json(directory / "config.json")["dtype"] == "bfloat16"
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert not model.training
    assert model.config.hidden_size == 32
    assert torch.get_num_threads() == CPU_THREADS


def test_load_model_refuses_a_parameter_that_is_not_float32(tiny_pretrained, monkeypatch):
    half = load_model(tiny_pretrained.directory(), "cpu").to(torch.bfloat16)
    monkeypatch.setattr(
        encoder_module, "AutoModel", SimpleNamespace(from_pretrained=lambda *args, **kwargs: half)
    )

    with pytest.raises(ValueError, match=r"dtypes \['torch.bfloat16'\]: it runs in float32"):
        load_model(tiny_pretrained.directory(), "cpu")


def test_encode_pools_the_end_token_of_each_text(tiny_pretrained, tiny_config):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    vectors = encoder.encode(TEXTS)

    for row, text in enumerate(TEXTS):
        expected = alone(encoder.model, encoder.tokenizer, text, tiny_config.max_length)
        np.testing.assert_allclose(vectors[row], expected, rtol=0, atol=1e-5)


def test_a_text_alone_equals_the_same_text_in_a_padded_batch(tiny_pretrained, tiny_config):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    together = encoder.encode(TEXTS)  # one batch, padded to 7 tokens

    for row, text in enumerate(TEXTS):
        np.testing.assert_allclose(together[row], encoder.encode([text])[0], rtol=0, atol=1e-5)


def test_encode_returns_distinct_unit_norm_float32_rows_in_input_order(
    tiny_pretrained, tiny_config
):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    vectors = encoder.encode([*TEXTS, LONG_TEXT])

    assert vectors.dtype == np.float32
    assert vectors.shape == (4, encoder.width) == (4, 32)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=0, atol=1e-6)
    assert len({tuple(row) for row in vectors.round(4)}) == 4
    reverse = encoder.encode([LONG_TEXT, *reversed(TEXTS)])
    np.testing.assert_allclose(reverse, vectors[::-1], rtol=0, atol=1e-5)


def test_encode_of_no_text_is_an_empty_float32_array(tiny_pretrained, tiny_config):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    vectors = encoder.encode([])

    assert (vectors.shape, vectors.dtype) == ((0, 32), np.float32)


def test_encode_sorts_texts_by_length_and_batches_them_by_padded_tokens(
    tiny_pretrained, tiny_config
):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)
    model = load_model(tiny_pretrained.directory(), "cpu")
    shapes = batch_shapes(model)
    texts = [words(4), words(2), words(20), words(2), words(4)]  # 5, 3, 16 (cut), 3, 5 tokens

    vectors = encode(texts, tokenizer, model, tiny_config, device="cpu", batch_tokens=10)

    # 2 x 3 tokens; then 2 x 5 (a third text would pad to 15 > 10); the longest alone (16 > 10).
    assert shapes == [(2, 3), (2, 5), (1, 16)]
    for row, text in enumerate(texts):
        expected = alone(model, tokenizer, text, tiny_config.max_length)
        np.testing.assert_allclose(vectors[row], expected, rtol=0, atol=1e-5)


def test_encode_caps_the_texts_of_a_batch(tiny_pretrained, tiny_config):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)
    model = load_model(tiny_pretrained.directory(), "cpu")
    shapes = batch_shapes(model)

    encode(
        [words(3)] * 5,
        tokenizer,
        model,
        tiny_config,
        device="cpu",
        batch_tokens=1000,
        max_batch_sequences=2,
    )

    assert shapes == [(2, 4), (2, 4), (1, 4)]


def test_the_encoder_uses_the_gpu_budget_on_the_gpu_and_its_own_on_the_cpu(
    tiny_pretrained, tiny_config, monkeypatch
):
    calls = []

    def spy(texts, tokenizer, model, config, **budget):  # pylint: disable=unused-argument
        calls.append(budget)

    monkeypatch.setattr(encoder_module, "encode", spy)
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")

    encoder.encode(TEXTS)
    dataclasses.replace(encoder, device="cuda").encode(TEXTS)

    assert calls == [
        {"device": "cpu", "batch_tokens": tiny_config.cpu_batch_tokens},
        {
            "device": "cuda",
            "batch_tokens": GPU_BATCH_TOKENS,
            "max_batch_sequences": GPU_MAX_BATCH_SEQUENCES,
        },
    ]


def test_encode_refuses_a_text_that_does_not_end_with_the_end_token(tiny_pretrained, tiny_config):
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    other_end = dataclasses.replace(tiny_config, eos_token_id=2)

    with pytest.raises(ValueError, match="does not end with the end-of-sequence token 2"):
        encode(TEXTS, encoder.tokenizer, encoder.model, other_end, device="cpu", batch_tokens=64)


def test_encode_refuses_non_finite_embeddings(tiny_pretrained, tiny_config):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)
    model = load_model(tiny_pretrained.directory(), "cpu")
    with torch.no_grad():
        model.norm.weight.fill_(float("nan"))

    with pytest.raises(ValueError, match="non-finite embeddings"):
        encode(TEXTS, tokenizer, model, tiny_config, device="cpu", batch_tokens=64)


def test_token_counts_count_the_end_token_without_truncation(tiny_pretrained, tiny_config):
    tokenizer = load_tokenizer(tiny_pretrained.directory(), tiny_config)

    counts = token_counts([*TEXTS, LONG_TEXT], tokenizer)

    assert counts.dtype == np.int64
    assert counts.tolist() == [4, 7, 2, 31]
    assert token_counts([], tokenizer).shape == (0,)


def test_the_frozen_encoder_is_checked_and_loaded_once_per_folder_and_device(
    tiny_pretrained, tiny_config, monkeypatch
):
    checked = []

    def check(record):
        checked.append(record)
        return check_pretrained(record)

    monkeypatch.setattr(encoder_module, "check_pretrained", check)
    first = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    shorter = frozen_encoder(tiny_pretrained, dataclasses.replace(tiny_config, max_length=8), "cpu")

    assert shorter.model is first.model
    assert shorter.tokenizer is first.tokenizer
    assert (first.config.max_length, shorter.config.max_length) == (16, 8)
    assert isinstance(first, Encoder) and first.device == "cpu"
    assert len(checked) == 1
    clear_memo()
    assert frozen_encoder(tiny_pretrained, tiny_config, "cpu").model is not first.model
    assert len(checked) == 2


def test_the_frozen_encoder_refuses_changed_weights_or_another_end_token(
    tiny_pretrained, tiny_config
):
    frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    with pytest.raises(ValueError, match="end-of-sequence token is 1, but encoder.json says 2"):
        frozen_encoder(tiny_pretrained, dataclasses.replace(tiny_config, eos_token_id=2), "cpu")

    clear_memo()
    (tiny_pretrained.directory() / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"changed \['config.json'\]"):
        frozen_encoder(tiny_pretrained, tiny_config, "cpu")
