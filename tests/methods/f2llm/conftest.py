"""Fixtures of the F2LLM tests: a tiny random encoder, saved as a pinned pretrained repository.

Nothing is downloaded: the encoder is a random ``Qwen3Model`` built from a config (2 layers,
hidden size 32), and its tokenizer a word-level ``tokenizers`` tokenizer over ``TINY_WORDS``
that, like F2LLM's, pads with ``<|endoftext|>`` on the right and appends ``<|im_end|>`` to every
input. The command tests (``test_main.py``, ``test_main_checks.py``) share the dataset, split,
folders and tuning results their commands run on, last in this file.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from codeforces_tags.cache import ArrayCache
from codeforces_tags.config import (
    DEVICE_ENV_VAR,
    FOCUS_TAGS,
    LEGACY_DEVICE_ENV_VAR,
    PRETRAINED_DIR_ENV_VAR,
    RAW_INPUT_COLUMNS,
)
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.methods import METHODS
from codeforces_tags.methods.f2llm import __main__ as cli
from codeforces_tags.methods.f2llm.embedding import INPUTS, cached_feature_frame, feature_frame
from codeforces_tags.methods.f2llm.encoder import (
    EncoderConfig,
    PretrainedRecord,
    clear_memo,
    repository_files,
    write_json,
)
from codeforces_tags.methods.f2llm.estimators import make_estimator, tfidf_params
from codeforces_tags.methods.f2llm.settings import (
    EARLY_FUSION,
    EARLY_FUSION_GRID,
    LATE_FUSION,
    LATE_FUSION_GRID,
    NAMES,
    POOLING,
    PROBE,
)
from codeforces_tags.methods.f2llm.tuning import (
    JOURNAL_FILE,
    OUT_OF_FOLD_FILE,
    PARAMS_FILE,
    TRIALS_FILE,
    TUNING_FOLDER,
    cross_fitted_macro_f1,
    load_params,
)
from codeforces_tags.methods.parts import read_parts
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.pretrained import snapshot_folder

# The tiny tokenizer's special tokens, F2LLM's: <|endoftext|> pads, <|im_end|> ends every input.
TINY_PAD = "<|endoftext|>"
TINY_EOS = "<|im_end|>"
TINY_UNK = "[UNK]"
# The words the tiny tokenizer knows, lower-cased; any other word is TINY_UNK. The synthetic
# problems are written with them, so that their embeddings differ.
TINY_WORDS = (
    "instruct",
    "classify",
    "the",
    "problem",
    "python",
    "solution",
    "query",
    "given",
    "a",
    "of",
    "and",
    "find",
    "count",
    "print",
    "number",
    "integer",
    "sum",
    "tree",
    "vertex",
    "edge",
    "graph",
    "path",
    "string",
    "letter",
    "prime",
    "divisor",
    "gcd",
    "modulo",
    "point",
    "polygon",
    "circle",
    "area",
    "game",
    "player",
    "alice",
    "bob",
    "wins",
    "probability",
    "expected",
    "value",
    "array",
    "test",
    "cases",
    "n",
    "m",
    "s",
    "x",
    "int",
    "input",
    "for",
    "in",
    "range",
    "if",
    "return",
    "def",
)
TINY_VOCABULARY = {
    token: index for index, token in enumerate((TINY_PAD, TINY_EOS, TINY_UNK, *TINY_WORDS))
}
# Where the tiny repository is saved, under the test's weights root.
TINY_REPO_ID = "tiny/qwen3"
TINY_REVISION = "0" * 40
TINY_HIDDEN_SIZE = 32


def tiny_tokenizer():
    """Build the tiny tokenizer: words of ``TINY_WORDS``, ``TINY_EOS`` appended, right padding."""
    tokenizer = Tokenizer(models.WordLevel(vocab=TINY_VOCABULARY, unk_token=TINY_UNK))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"$A {TINY_EOS}", special_tokens=[(TINY_EOS, TINY_VOCABULARY[TINY_EOS])]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=TINY_EOS,
        pad_token=TINY_PAD,
        unk_token=TINY_UNK,
        padding_side="right",
    )


def tiny_model():
    """Build the tiny random encoder, seeded, in bfloat16 as F2LLM's weights are stored."""
    config = Qwen3Config(
        vocab_size=len(TINY_VOCABULARY),
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=TINY_VOCABULARY[TINY_PAD],
        eos_token_id=TINY_VOCABULARY[TINY_EOS],
    )
    torch.manual_seed(0)
    return Qwen3Model(config).to(torch.bfloat16)


@pytest.fixture(name="no_gpu", autouse=True)
def no_gpu_fixture(monkeypatch):
    """Run every test as on the CI machine, which has no GPU, even where torch sees one.

    A loaded method encodes on the GPU when torch sees one, so ``torch.cuda.is_available``
    answers False, and ``$CODEFORCES_DEVICE`` is "cpu" for the isolation subprocess of
    ``check_tagging_method``, which inherits the environment and not the monkeypatch. The
    variable the F2LLM methods read before, which ``requested_device`` refuses, is removed. A
    test of the automatic device fakes ``is_available`` and removes the variable itself.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)


@pytest.fixture(name="tiny_pretrained")
def tiny_pretrained_fixture(tmp_path, monkeypatch):
    """Save the tiny encoder and its tokenizer as a pinned repository; return its record.

    ``$CODEFORCES_PRETRAINED_DIR`` points at ``tmp_path / "pretrained"`` (the isolation
    subprocess of ``check_tagging_method`` inherits it), and the files go to the snapshot's
    folder there, ``snapshot_folder(TINY_REPO_ID, TINY_REVISION)``. The memo of frozen encoders
    is cleared before and after the test.
    """
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "pretrained"))
    directory = snapshot_folder(TINY_REPO_ID, TINY_REVISION)
    tiny_tokenizer().save_pretrained(directory)
    tiny_model().save_pretrained(directory)
    clear_memo()
    yield PretrainedRecord(TINY_REPO_ID, TINY_REVISION, repository_files(directory))
    clear_memo()


@pytest.fixture(name="tiny_config")
def tiny_config_fixture(tiny_pretrained):  # pylint: disable=unused-argument
    """Return the tiny encoder's configuration: its end-of-sequence id, 16 tokens per input."""
    return EncoderConfig(
        statement_instruction="instruct classify the problem query ",
        solution_instruction="instruct classify the python solution query ",
        max_length=16,
        pooling=POOLING,
        eos_token_id=TINY_VOCABULARY[TINY_EOS],
        cpu_batch_tokens=64,
    )


# The words of each focus tag in the synthetic problems, and of the problems without a focus
# tag: words of TINY_WORDS, so that each tag's problems read differently to the tiny encoder.
SYNTHETIC_TAG_WORDS = {
    "math": ("sum", "integer"),
    "graphs": ("graph", "edge"),
    "strings": ("string", "letter"),
    "number theory": ("prime", "divisor"),
    "trees": ("tree", "vertex"),
    "geometry": ("point", "circle"),
    "games": ("alice", "bob"),
    "probabilities": ("probability", "expected"),
}
SYNTHETIC_NO_FOCUS_WORDS = ("array", "value")
# 40 problems with a focus tag (5 folds x 8 tags), then 5 without (one per fold).
SYNTHETIC_FOCUS_PROBLEMS = 40
SYNTHETIC_PROBLEMS = 45


def _synthetic_tags(position):
    """Return the tags of synthetic problem ``position``: focus tags, then other tags."""
    if position >= SYNTHETIC_FOCUS_PROBLEMS:
        return ["implementation"]
    tags = [FOCUS_TAGS[position % len(FOCUS_TAGS)]]
    if position % 10 == 9:  # an odd position, so never a math problem already
        tags.append("math")
    if position % 3 == 0:
        tags.append("implementation")
    return tags


def _synthetic_problem(make_raw_record, position):
    """Return the raw record of synthetic problem ``position``.

    Its statement holds words of ``TINY_WORDS`` only (the notes lose their ``Note`` heading in
    preprocessing), among them one word chosen by position. That word opens the description: the
    tiny encoder cuts its input to ``max_length`` tokens, instruction included, and a word further
    on would be lost, so that the encoder could not tell the statements of one tag apart.
    """
    tags = _synthetic_tags(position)
    first, second = SYNTHETIC_TAG_WORDS.get(tags[0], SYNTHETIC_NO_FOCUS_WORDS)
    word = TINY_WORDS[position % len(TINY_WORDS)]
    return make_raw_record(
        prob_desc_description=(
            f"{word} given a {first} and a {second} find the {first} of the {second}"
        ),
        prob_desc_input_spec=f"the {second} of the {first}",
        prob_desc_output_spec=f"print the {first}",
        prob_desc_notes=f"NoteThe {first} of the {second} and the {word}",
        source_code=(
            f"{first} = int(input())\nfor x in range({position}):\n    print({first} + {second})\n"
        ),
        tags=tags,
    )


@pytest.fixture(name="synthetic_raw")
def synthetic_raw_fixture(make_raw_record):
    """Return 45 raw problems as ``select_part(read_samples(), split, "dev")`` gives them.

    Problem ``position`` (0-44) sits in fold ``position % 5``. The first 40 have the focus tag
    ``FOCUS_TAGS[position % 8]``: as 5 and 8 are coprime, every focus tag has one such problem
    in every fold, so every training fold set, and the refit rows, hold positives and negatives
    of every tag. Positions 9, 19, 29 and 39 are also tagged math, and every third position
    implementation; positions 40-44 have no focus tag. Positions 5 and 6 share the
    near-duplicate group of positions 0 and 1 (the same folds); every other problem has its own.
    Sample ids are 100-144, out of order.
    """
    positions = range(SYNTHETIC_PROBLEMS)
    sample_ids = [100 + (7 * position) % SYNTHETIC_PROBLEMS for position in positions]
    records = [_synthetic_problem(make_raw_record, position) for position in positions]
    raw = pd.DataFrame(records, index=pd.Index(sample_ids, name="sample_id"))
    raw["fold"] = [position % 5 for position in positions]
    raw["group"] = [{5: 0, 6: 1}.get(position, position) for position in positions]
    return raw


@pytest.fixture(name="synthetic_labels")
def synthetic_labels_fixture(synthetic_raw):
    """Return the focus labels of ``synthetic_raw``: int 0/1, one column per focus tag."""
    return focus_labels(raw_to_problems(synthetic_raw))


@pytest.fixture(name="embedding_width")
def embedding_width_fixture():
    """Return the width of the synthetic embeddings."""
    return 8


@pytest.fixture(name="embedding_frame")
def embedding_frame_fixture(synthetic_raw, synthetic_labels, embedding_width):
    """Return the feature frame of the synthetic problems, with embeddings that carry the tags.

    Each input's vector is the sum of one random direction per focus tag the problem has, plus
    Gaussian noise (seeded), so that every head can learn every tag. No encoder runs.
    """
    rng = np.random.default_rng(0)
    labels = synthetic_labels.to_numpy(dtype=float)
    vectors = {
        name: labels @ rng.normal(size=(labels.shape[1], embedding_width))
        + 0.3 * rng.normal(size=(len(labels), embedding_width))
        for name in INPUTS
    }
    return feature_frame(synthetic_raw, vectors)


@pytest.fixture(name="small_tfidf")
def small_tfidf_fixture(make_params):
    """Return a linear-SVM TF-IDF configuration that keeps every term of the tiny vocabulary.

    Built by ``make_params`` (tests/conftest.py), on the frozen TF-IDF's blocks: statement and code.
    """
    return make_params(
        classifier="linear_svm",
        include_notes=True,
        statement_sublinear_tf=True,
        cue_weight=0.0,
        blocks=("statement", "code"),
    )


# The command tests run the commands on the synthetic problems, the dev part, and on copies of the
# first TEST_PROBLEMS of them, the test part, under sample ids (and groups) shifted by
# TEST_ID_OFFSET. A command-test module requests calls for each of its tests, and defines
# SCORE_ATOL, the score_atol of the F2LLM lines in the registry its commands read (not the
# measured one), and PROBE_PARAMS, the probe configuration that probe_tuned exports.
TEST_PROBLEMS = 4
TEST_ID_OFFSET = 10_000


@pytest.fixture(name="dataset")
def dataset_fixture(synthetic_raw, write_samples, make_raw_record):
    """Write the synthetic problems, then copies of the first ones as test problems."""
    records = {
        sample_id: make_raw_record(
            **{column: row[column] for column in RAW_INPUT_COLUMNS}, tags=list(row["tags"])
        )
        for sample_id, row in synthetic_raw.iterrows()
    }
    for sample_id in synthetic_raw.index[:TEST_PROBLEMS]:
        records[sample_id + TEST_ID_OFFSET] = records[sample_id]
    return write_samples(records)


@pytest.fixture(name="split_file")
def split_file_fixture(synthetic_raw, tmp_path):
    """Write the frozen split: the synthetic problems' folds and groups, then the test part."""
    test_ids = list(synthetic_raw.index[:TEST_PROBLEMS] + TEST_ID_OFFSET)
    split = pd.DataFrame(
        {
            "part": ["dev"] * len(synthetic_raw) + ["test"] * TEST_PROBLEMS,
            "fold": [*synthetic_raw["fold"], *[-1] * TEST_PROBLEMS],
            "group": [*synthetic_raw["group"], *test_ids],
        },
        index=pd.Index([*synthetic_raw.index, *test_ids], name="sample_id"),
    )
    path = tmp_path / "split.csv"
    split.to_csv(path)
    return path


@pytest.fixture(name="common")
def common_fixture(dataset, split_file):
    """Return the dataset and split arguments every command but fetch takes."""
    return ["--data", str(dataset), "--split", str(split_file)]


@pytest.fixture(name="calls")
def calls_fixture(request, monkeypatch, tmp_path, tiny_pretrained, tiny_config):
    """Point the commands at the tiny encoder and at folders under tmp_path.

    The dataset check, which only the challenge dataset passes, and the MLflow log are replaced;
    returns the arguments each was called with. The F2LLM lines of the registry the commands
    read hold the requesting module's ``SCORE_ATOL``. The MLflow test puts the real log back.
    """
    calls = {"check_dataset": [], "log_method": []}

    def log(folder, record, metrics, *, split, **options):
        calls["log_method"].append((folder, record, metrics, split, options))
        return "model-id"

    monkeypatch.setattr(cli, "_encoding", lambda: (tiny_pretrained, tiny_config))
    monkeypatch.setattr(cli, "open_cache", lambda: ArrayCache(tmp_path / "cache"))
    monkeypatch.setattr(cli, "SHARED_ASSETS_DIR", tmp_path / "shared")
    monkeypatch.setattr(
        cli, "journal_path", lambda name: tmp_path / "experiments" / name / JOURNAL_FILE
    )
    monkeypatch.setattr(
        cli, "tuning_dir", lambda name: tmp_path / "experiments" / name / TUNING_FOLDER
    )
    monkeypatch.setattr(cli, "check_dataset", calls["check_dataset"].append)
    monkeypatch.setattr(cli, "log_method", log)
    score_atol = request.module.SCORE_ATOL
    tiny = {name: dataclasses.replace(METHODS[name], score_atol=score_atol) for name in NAMES}
    monkeypatch.setattr(cli, "METHODS", {**METHODS, **tiny})
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    return calls


@pytest.fixture(name="embedded")
def embedded_fixture(common, tmp_path):
    """Embed every problem into the test's cache with the tiny encoder; return the cache."""
    cli.main(["embed", "--device", "cpu", *common])
    return tmp_path / "cache"


@pytest.fixture(name="write_tuning")
def write_tuning_fixture(common, embedded, tmp_path, tiny_pretrained, tiny_config):
    """Return a writer of a tuned method's tuning/ folder in the test's experiments folder.

    ``write_tuning(name, params)`` writes, over any earlier export, what export writes when the
    study chose ``params``: ``params.json``, a ``trials.csv`` of one made-up trial, and
    ``out_of_fold.csv``, the cross-fitted out-of-fold F1 of ``params`` on the dev problems as the
    cache embeds them then (late fusion fusing the probe configuration exported there, as tune
    and export do). It returns the folder.
    """
    parts = read_parts(common[1], common[3])  # --data and --split
    experiments = tmp_path / "experiments"

    def write(name, params):
        frame = cached_feature_frame(parts.dev, ArrayCache(embedded), tiny_pretrained, tiny_config)
        labels = parts.labels.loc[frame.index]
        estimator = make_estimator(
            name,
            params,
            width=TINY_HIDDEN_SIZE,
            probe_params=(
                load_params(PROBE, experiments / PROBE / TUNING_FOLDER)
                if name == LATE_FUSION
                else None
            ),
            tfidf=None if name == PROBE else tfidf_params(),
        )
        _, table = cross_fitted_macro_f1(estimator, frame, labels, frame["fold"])
        folder = experiments / name / TUNING_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / PARAMS_FILE, params)
        (folder / TRIALS_FILE).write_text("number,value\n0,0.5\n", encoding="utf-8")
        table.to_csv(folder / OUT_OF_FOLD_FILE, index_label="metric")
        return folder

    return write


@pytest.fixture(name="probe_tuned")
def probe_tuned_fixture(request, write_tuning):
    """Write the probe's tuning/ folder, choosing the requesting module's PROBE_PARAMS."""
    return write_tuning(PROBE, request.module.PROBE_PARAMS)


@pytest.fixture(name="fusions_tuned")
def fusions_tuned_fixture(write_tuning, probe_tuned):  # pylint: disable=unused-argument
    """Write the two fusion methods' tuning/ folders, choosing their first grid point."""
    write_tuning(LATE_FUSION, LATE_FUSION_GRID[0])
    write_tuning(EARLY_FUSION, EARLY_FUSION_GRID[0])


@pytest.fixture(name="assets")
def assets_fixture(tmp_path):
    """Return the folder of the frozen methods; nothing is frozen in it yet."""
    return tmp_path / "assets"


@pytest.fixture(name="freeze_args")
def freeze_args_fixture(common, assets, store):
    """Return the probe's freeze arguments, on CPU; the tree may be dirty while the tests run."""
    options = ["--device", "cpu", "--assets", str(assets), "--allow-dirty"]
    options += ["--tracking-uri", store["tracking_uri"]]
    options += ["--artifact-dir", str(store["artifact_dir"])]
    return ["freeze", "--method", PROBE, *common, *options]


@pytest.fixture(name="verify_args")
def verify_args_fixture(common, assets):
    """Return the probe's verify arguments (on CPU, verify's default)."""
    return ["verify", "--method", PROBE, *common, "--assets", str(assets)]


@pytest.fixture(name="frozen")
def frozen_fixture(freeze_args, assets, embedded, probe_tuned):  # pylint: disable=unused-argument
    """Freeze the probe from the embedded cache and its tuning/ folder; return its directory."""
    cli.main(freeze_args)
    return assets / PROBE
