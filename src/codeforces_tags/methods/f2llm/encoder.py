"""The F2LLM encoder: its weights, its configuration, loading it and encoding.

- Weights: ``fetch`` downloads the pinned repository files into ``PretrainedRecord.directory()``,
  the snapshot's folder in the pretrained cache (``codeforces_tags.pretrained.snapshot_folder``,
  read at call time and shared with the ModernBERT methods), and checks them;
  ``check_pretrained`` checks them again before they are loaded. A method folder holds
  ``pretrained.json`` (a ``PretrainedRecord``), never the weights.
- ``EncoderConfig``: every encoder constant a method encodes with, saved in its method folder as
  ``encoder.json``, so that a loaded method reads none of them from ``settings``.
- Loading: ``load_tokenizer`` and ``load_model`` (float32, local safetensors only), and
  ``frozen_encoder``, which loads the frozen encoder of a folder and a device once per process.
- Encoding: ``encode`` (last-token pooling, L2-normalised, float32; texts sorted by length and
  batched by padded tokens), ``Encoder.encode`` (with its device's batch budget) and
  ``token_counts``.
- Devices: ``resolve_device``, and ``loaded_device``, the device a loaded method encodes on
  (``$CODEFORCES_DEVICE``, read by ``codeforces_tags.devices.requested_device``).

This module is on the methods' prediction path: it imports torch and transformers, never
mlflow, optuna or openai, and imports huggingface_hub only inside ``fetch``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from codeforces_tags.devices import requested_device
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
    PICKLE_SUFFIXES,
    POOLING,
    SOLUTION_INSTRUCTION,
    STATEMENT_INSTRUCTION,
)
from codeforces_tags.pretrained import snapshot_folder
from codeforces_tags.provenance import file_sha256

# The files of a method folder that say what it encodes with: the encoder constants, and the
# pinned weights it loads from the pretrained cache.
ENCODER_FILE = "encoder.json"
PRETRAINED_FILE = "pretrained.json"
# The devices the encoder runs on.
_DEVICES = ("cpu", "cuda")
# The folder where snapshot_download keeps its own metadata (timestamps): not a repository file.
_DOWNLOAD_CACHE = ".cache"
# The command that downloads the pinned weights, named by the errors that need it.
_FETCH_COMMAND = "uv run --group f2llm python -m codeforces_tags.methods.f2llm fetch"


@dataclass(frozen=True)
class EncoderConfig:
    """Every constant the encoder encodes with: ``encoder.json`` in a method folder.

    Attributes:
        statement_instruction: Prepended to the statement input.
        solution_instruction: Prepended to the solution input.
        max_length: Tokens per encoder input, the end-of-sequence token included; a longer
            input loses its end.
        pooling: What is stored; ``POOLING`` (last-token pooling, L2-normalised, float32) is the
            only pooling this code implements.
        eos_token_id: The token the tokenizer appends to every input; its last hidden state,
            normalised, is the embedding.
        cpu_batch_tokens: Padded tokens per encoder batch on CPU.
    """

    statement_instruction: str
    solution_instruction: str
    max_length: int
    pooling: str
    eos_token_id: int
    cpu_batch_tokens: int

    def __post_init__(self) -> None:
        """Refuse a pooling this code does not implement.

        Raises:
            ValueError: If ``pooling`` is not ``POOLING``.
        """
        if self.pooling != POOLING:
            raise ValueError(
                f"The pooling {self.pooling!r} is not implemented: the encoder stores {POOLING!r}."
            )

    @classmethod
    def from_settings(cls) -> EncoderConfig:
        """Build the configuration the fit path encodes with, from ``settings``.

        Returns:
            The configuration of the package's settings.
        """
        return cls(
            statement_instruction=STATEMENT_INSTRUCTION,
            solution_instruction=SOLUTION_INSTRUCTION,
            max_length=MAX_LENGTH,
            pooling=POOLING,
            eos_token_id=EOS_TOKEN_ID,
            cpu_batch_tokens=CPU_BATCH_TOKENS,
        )

    def to_dict(self) -> dict:
        """Return the fields, as ``encoder.json`` stores them.

        Returns:
            ``{field name: value}``.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> EncoderConfig:
        """Rebuild a configuration from ``to_dict``'s output.

        Args:
            values: Every field, and nothing else.

        Returns:
            The configuration.

        Raises:
            ValueError: If a field is missing, a key is not a field, or the pooling is not
                ``POOLING``.
        """
        _check_keys(values, [field.name for field in fields(cls)], ENCODER_FILE)
        return cls(**values)


@dataclass(frozen=True)
class PretrainedRecord:
    """The pretrained weights a method encodes with: ``pretrained.json`` in a method folder.

    Attributes:
        repo_id: The Hugging Face repository, e.g. ``"codefuse-ai/F2LLM-0.6B"``.
        revision: The repository's commit.
        files: Each repository file's path (relative, in POSIX form) and its sha256.
    """

    repo_id: str
    revision: str
    files: dict[str, str]

    def __post_init__(self) -> None:
        """Refuse a name that would leave the folder of the weights.

        Raises:
            ValueError: If the repository, the revision or a file path is empty, absolute or
                holds ``..``.
        """
        for name in (self.repo_id, self.revision, *self.files):
            _check_relative(name)

    @classmethod
    def from_settings(cls) -> PretrainedRecord:
        """Build the record of the pinned encoder, from ``settings``.

        Returns:
            ``ENCODER_REPO_ID`` at ``ENCODER_REVISION``, with the files of ``ENCODER_FILES``.
        """
        return cls(ENCODER_REPO_ID, ENCODER_REVISION, dict(ENCODER_FILES))

    def to_dict(self) -> dict:
        """Return the record as ``pretrained.json`` stores it.

        Returns:
            ``{"repo_id", "revision", "files"}``, ``files`` being a list of
            ``{"path", "sha256"}`` sorted by path.
        """
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "files": [
                {"path": path, "sha256": sha256} for path, sha256 in sorted(self.files.items())
            ],
        }

    @classmethod
    def from_dict(cls, values: dict) -> PretrainedRecord:
        """Rebuild a record from ``to_dict``'s output.

        Args:
            values: ``{"repo_id", "revision", "files"}``, as ``to_dict`` returns them.

        Returns:
            The record.

        Raises:
            ValueError: If a key is missing or unexpected, a path repeats, or a name would
                leave the folder of the weights.
        """
        _check_keys(values, ("repo_id", "revision", "files"), PRETRAINED_FILE)
        for entry in values["files"]:
            _check_keys(entry, ("path", "sha256"), f"a file of {PRETRAINED_FILE}")
        files = {entry["path"]: entry["sha256"] for entry in values["files"]}
        if len(files) != len(values["files"]):
            raise ValueError(f"{PRETRAINED_FILE} lists a file twice: each path is listed once.")
        return cls(values["repo_id"], values["revision"], files)

    def directory(self) -> Path:
        """Return the folder of the weights.

        Returns:
            ``snapshot_folder(repo_id, revision)``: ``<cache>/<org>--<name>/<revision>`` in the
            pretrained cache, read at call time.

        Raises:
            ValueError: If the repository or the revision would not give one folder name each
                (``snapshot_folder`` says which).
        """
        return snapshot_folder(self.repo_id, self.revision)


def repository_files(directory: Path) -> dict[str, str]:
    """Hash the repository files of a downloaded folder.

    Args:
        directory: The folder ``snapshot_download`` wrote.

    Returns:
        ``{path relative to directory, in POSIX form: sha256}`` for every file, sorted by path,
        without the ``.cache/`` folder that ``snapshot_download`` writes its metadata to.
    """
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).parts[0] != _DOWNLOAD_CACHE
    }


def check_no_pickle_files(directory: Path) -> None:
    """Refuse a folder that holds a file of a pickle format, ``.cache/`` included.

    Args:
        directory: The folder of the weights.

    Raises:
        ValueError: If a file's suffix is one of ``PICKLE_SUFFIXES``, whatever its case.
    """
    directory = Path(directory)
    found = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.suffix.lower() in PICKLE_SUFFIXES
    )
    if found:
        raise ValueError(
            f"{directory} holds files of pickle formats, which run code when loaded: {found}. "
            "Delete them: the encoder loads safetensors only."
        )


def check_pretrained(record: PretrainedRecord) -> Path:
    """Check the weights of a record before loading them.

    Args:
        record: The weights to check.

    Returns:
        ``record.directory()``, the folder to load them from.

    Raises:
        ValueError: If a listed file is missing or has another sha256, or if the folder holds
            a file of a pickle format.
    """
    directory = record.directory()
    check_no_pickle_files(directory)
    found = {
        path: file_sha256(directory / path) for path in record.files if (directory / path).is_file()
    }
    _check_files(directory, found, record.files)
    return directory


def fetch(
    record: PretrainedRecord | None = None, *, download: Callable[..., object] | None = None
) -> Path:
    """Download the pinned repository files, then check them.

    Args:
        record: The weights to download; ``PretrainedRecord.from_settings()`` when None.
        download: Called as ``download(repo_id=, revision=, allow_patterns=, local_dir=)``;
            ``huggingface_hub.snapshot_download`` when None (the tests pass a fake).

    Returns:
        The folder of the weights, ``record.directory()``.

    Raises:
        ValueError: If the folder holds a file of a pickle format, or if its repository files
            (without ``.cache/``) are not exactly the record's, with the same sha256.
    """
    if record is None:
        record = PretrainedRecord.from_settings()
    if download is None:
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

        download = snapshot_download
    directory = record.directory()
    download(
        repo_id=record.repo_id,
        revision=record.revision,
        allow_patterns=list(ENCODER_FILE_PATTERNS),
        local_dir=directory,
    )
    check_no_pickle_files(directory)
    _check_files(directory, repository_files(directory), record.files)
    return directory


def write_json(path: Path, values: dict) -> None:
    """Write a JSON file: sorted keys, an indent of 2 and a trailing newline.

    Args:
        path: The file.
        values: JSON types only (no NaN or infinity).
    """
    text = json.dumps(values, indent=2, sort_keys=True, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    """Read a JSON file that ``write_json`` wrote.

    Args:
        path: The file.

    Returns:
        Its values.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_device(requested: str) -> str:
    """Check the device to encode on.

    Args:
        requested: ``"cpu"``, ``"cuda"``, or ``"auto"`` (``AUTO_DEVICE``).

    Returns:
        ``requested``; for ``"auto"``, ``"cuda"`` when torch sees a GPU, else ``"cpu"``.

    Raises:
        ValueError: If ``requested`` is none of the three.
        RuntimeError: If it is ``"cuda"`` and torch sees no GPU.
    """
    if requested == AUTO_DEVICE:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in _DEVICES:
        raise ValueError(f"{requested!r} is not a device: expected one of {_DEVICES}.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The device 'cuda' was asked for, but torch sees no GPU.")
    return requested


def loaded_device() -> str:
    """Return the device a loaded method encodes on, read at call time.

    Returns:
        ``$CODEFORCES_DEVICE`` when it names ``"cpu"`` or ``"cuda"`` (``requested_device``),
        checked by ``resolve_device``. Unset, empty or ``"auto"`` (``AUTO_DEVICE``), ``"cuda"``
        when torch sees a GPU, else ``"cpu"`` (a user may have no GPU).

    Raises:
        ValueError: If the variable names no device setting, or if the variable the F2LLM
            methods read before is set (``requested_device`` names both).
        RuntimeError: If it names ``"cuda"`` and torch sees no GPU.
    """
    return resolve_device(requested_device())


def load_tokenizer(directory: Path, config: EncoderConfig):
    """Load the encoder's tokenizer from local files only.

    Args:
        directory: The folder of the weights, as ``check_pretrained`` returns it.
        config: The encoder configuration, whose ``eos_token_id`` the tokenizer must append.

    Returns:
        The tokenizer.

    Raises:
        ValueError: If it pads on the left, or if its end-of-sequence token is not
            ``config.eos_token_id``: pooling reads the end-of-sequence token at the last real
            position, ``attention_mask.sum(1) - 1``, which holds under right padding only.
    """
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    _check_tokenizer(tokenizer, config)
    return tokenizer


def load_model(directory: Path, device: str) -> torch.nn.Module:
    """Load the encoder in float32, for inference, from local safetensors only.

    The stored weights are bfloat16: ``dtype=torch.float32`` upcasts them, since transformers
    would otherwise keep the stored dtype. On CPU, torch then uses ``CPU_THREADS`` threads.

    Args:
        directory: The folder of the weights, as ``check_pretrained`` returns it.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The model, in eval mode, on ``device``.

    Raises:
        ValueError: If a parameter is not float32.
    """
    model = AutoModel.from_pretrained(
        directory,
        dtype=torch.float32,
        attn_implementation="sdpa",
        use_safetensors=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if dtypes != [str(torch.float32)]:
        raise ValueError(f"The encoder holds parameters of dtypes {dtypes}: it runs in float32.")
    if device == "cpu":
        torch.set_num_threads(CPU_THREADS)
    return model.to(device).eval()


@dataclass
class Encoder:
    """A loaded encoder, with the configuration and the device it encodes with.

    Attributes:
        tokenizer: The tokenizer, as ``load_tokenizer`` returns it.
        model: The encoder, as ``load_model`` returns it.
        config: The encoder configuration.
        device: ``"cpu"`` or ``"cuda"``, where ``model`` is.
    """

    tokenizer: object
    model: torch.nn.Module
    config: EncoderConfig
    device: str

    @property
    def width(self) -> int:
        """The embedding width: the model's hidden size."""
        return int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts with the batch budget of the encoder's device.

        On CPU, batches hold ``config.cpu_batch_tokens`` padded tokens; on the GPU,
        ``GPU_BATCH_TOKENS`` and at most ``GPU_MAX_BATCH_SEQUENCES`` texts.

        Args:
            texts: The encoder inputs.

        Returns:
            Float32 ``(len(texts), width)``, one unit-norm row per text, in input order.
        """
        if self.device == "cuda":
            budget = {
                "batch_tokens": GPU_BATCH_TOKENS,
                "max_batch_sequences": GPU_MAX_BATCH_SEQUENCES,
            }
        else:
            budget = {"batch_tokens": self.config.cpu_batch_tokens}
        return encode(texts, self.tokenizer, self.model, self.config, device=self.device, **budget)


def encode(  # pylint: disable=too-many-arguments
    texts: Sequence[str],
    tokenizer,
    model: torch.nn.Module,
    config: EncoderConfig,
    *,
    device: str,
    batch_tokens: int,
    max_batch_sequences: int | None = None,
) -> np.ndarray:
    """Embed texts: the last hidden state of their end-of-sequence token, L2-normalised.

    Each text is truncated to ``config.max_length`` tokens (the tokenizer keeps its
    end-of-sequence token last). The texts are sorted by token count and cut into batches that
    pad to at most ``batch_tokens`` tokens (a longer text is a batch of its own) and hold at
    most ``max_batch_sequences`` texts, then returned in input order. Under right padding a
    text's embedding does not depend on its batch, up to float rounding.

    Args:
        texts: The encoder inputs.
        tokenizer: The tokenizer, as ``load_tokenizer`` returns it.
        model: The encoder, as ``load_model`` returns it, on ``device``.
        config: The encoder configuration.
        device: Where ``model`` is: ``"cpu"`` or ``"cuda"``.
        batch_tokens: Padded tokens per batch, at most.
        max_batch_sequences: Texts per batch, at most; None for no cap.

    Returns:
        Float32 ``(len(texts), hidden_size)``, one unit-norm row per text, in input order.

    Raises:
        ValueError: If a text's last real token is not ``config.eos_token_id``, or if an
            embedding is not finite.
    """
    texts = list(texts)
    vectors = np.zeros((len(texts), int(model.config.hidden_size)), dtype=np.float32)
    if not texts:
        return vectors
    lengths = np.minimum(token_counts(texts, tokenizer), config.max_length)
    order = np.argsort(lengths, kind="stable")
    for batch in _batches(lengths[order], batch_tokens, max_batch_sequences):
        rows = order[batch]
        batch_texts = [texts[row] for row in rows]
        vectors[rows] = _encode_batch(batch_texts, tokenizer, model, config, device)
    return vectors


def token_counts(texts: Sequence[str], tokenizer) -> np.ndarray:
    """Count each text's tokens, its end-of-sequence token included, without truncation.

    Args:
        texts: The encoder inputs.
        tokenizer: The tokenizer, as ``load_tokenizer`` returns it.

    Returns:
        Int64 ``(len(texts),)``.
    """
    texts = list(texts)
    if not texts:
        return np.zeros(0, dtype=np.int64)
    input_ids = tokenizer(texts, truncation=False)["input_ids"]
    return np.array([len(ids) for ids in input_ids], dtype=np.int64)


# The frozen encoders loaded so far, as (tokenizer, model) by (folder of the weights, device):
# each is loaded, and its weights checked, once per process. They hold no adapter, and are never
# handed to training: peft injects its layers into the model it is given, in place.
_MEMO: dict[tuple[Path, str], tuple[object, torch.nn.Module]] = {}


def frozen_encoder(record: PretrainedRecord, config: EncoderConfig, device: str) -> Encoder:
    """Return the frozen encoder of a record on a device, loaded once per process.

    The first call for a folder and a device checks the weights (``check_pretrained``) and
    loads them; later calls share the loaded tokenizer and model, with their own
    configuration. Training never uses it: SetFit loads its own copy with ``load_model``.

    Args:
        record: The pinned weights.
        config: The encoder configuration to encode with.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The encoder.

    Raises:
        ValueError: If the weights are not the pinned ones, or the tokenizer does not match
            ``config``.
        RuntimeError: If ``device`` is ``"cuda"`` and torch sees no GPU.
    """
    device = resolve_device(device)
    key = (record.directory(), device)
    if key not in _MEMO:
        directory = check_pretrained(record)
        _MEMO[key] = (load_tokenizer(directory, config), load_model(directory, device))
    tokenizer, model = _MEMO[key]
    _check_tokenizer(tokenizer, config)
    return Encoder(tokenizer, model, config, device)


def clear_memo() -> None:
    """Forget every frozen encoder ``frozen_encoder`` loaded (the tests call it)."""
    _MEMO.clear()


def _check_keys(values: dict, expected: Sequence[str], what: str) -> None:
    """Refuse a mapping whose keys are not exactly ``expected``."""
    missing = sorted(set(expected) - set(values))
    unexpected = sorted(set(values) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{what} takes exactly the keys {sorted(expected)}: missing {missing}, "
            f"unexpected {unexpected}."
        )


def _check_relative(name: str) -> None:
    """Refuse a name that is empty, absolute or climbs out of its folder with ``..``."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name!r} is not a relative path inside the folder of the weights.")


def _check_files(directory: Path, found: dict[str, str], expected: dict[str, str]) -> None:
    """Refuse files that are not exactly the expected ones, with their sha256."""
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    changed = sorted(path for path in set(found) & set(expected) if found[path] != expected[path])
    problems = [
        f"{what} {paths}"
        for what, paths in (("missing", missing), ("not pinned", unexpected), ("changed", changed))
        if paths
    ]
    if problems:
        raise ValueError(
            f"{directory} does not hold the pinned encoder files ({'; '.join(problems)}): "
            f"delete the folder, then run `{_FETCH_COMMAND}`."
        )


def _check_tokenizer(tokenizer, config: EncoderConfig) -> None:
    """Refuse a tokenizer that pads on the left or ends inputs with another token."""
    if tokenizer.padding_side != "right":
        raise ValueError(
            f"The tokenizer pads on the {tokenizer.padding_side}: last-token pooling needs "
            "right padding."
        )
    if tokenizer.eos_token_id != config.eos_token_id:
        raise ValueError(
            f"The tokenizer's end-of-sequence token is {tokenizer.eos_token_id}, but "
            f"{ENCODER_FILE} says {config.eos_token_id}."
        )


def _batches(lengths: np.ndarray, batch_tokens: int, max_sequences: int | None) -> list[slice]:
    """Cut texts sorted by token count into consecutive batches within the budget.

    A batch pads every text to its last, longest one: it holds count x last length tokens.
    """
    limit = max_sequences or len(lengths)
    batches = []
    start = 0
    for end, length in enumerate(lengths):
        count = end - start + 1
        if end > start and (count * int(length) > batch_tokens or count > limit):
            batches.append(slice(start, end))
            start = end
    batches.append(slice(start, len(lengths)))
    return batches


def _encode_batch(
    texts: list[str], tokenizer, model: torch.nn.Module, config: EncoderConfig, device: str
) -> np.ndarray:
    """Embed one batch: right-padded, the end-of-sequence token's state, L2-normalised."""
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt"
    ).to(device)
    rows = torch.arange(len(texts), device=device)
    last = tokens["attention_mask"].sum(dim=1) - 1
    if not bool((tokens["input_ids"][rows, last] == config.eos_token_id).all()):
        raise ValueError(
            f"A text does not end with the end-of-sequence token {config.eos_token_id}: "
            "the pooling reads its hidden state."
        )
    with torch.inference_mode():
        hidden = model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        ).last_hidden_state
    pooled = torch.nn.functional.normalize(hidden[rows, last], dim=1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("The encoder returned non-finite embeddings.")
    return pooled.cpu().numpy()
