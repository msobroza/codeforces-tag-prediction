"""Settings of the F2LLM embedding methods: what their recipes hash (docs/adr/0004).

The values that fix the encoder, its inputs, the grids of the three tuned methods and SetFit's
hyperparameters live here; project-wide values (the focus tags, the cache and experiments
folders, the seeds, the tolerance of the exports) come from ``codeforces_tags.config`` and are
never copied. ``SETTINGS`` is the part of them that a method's ``recipe_sha256`` hashes.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from codeforces_tags import config
from codeforces_tags.provenance import sha256_json

# The registry keys of the four methods. NAMES is in registry order (sorted by name); TUNED is in
# tuning order: late fusion and SetFit read the probe's tuned settings, so the probe comes first.
PROBE = "f2llm_probe"
LATE_FUSION = "f2llm_late_fusion"
EARLY_FUSION = "f2llm_early_fusion"
SETFIT_METHOD = "f2llm_setfit"
NAMES: tuple[str, ...] = (EARLY_FUSION, LATE_FUSION, PROBE, SETFIT_METHOD)
TUNED: tuple[str, ...] = (PROBE, LATE_FUSION, EARLY_FUSION)

# The encoder: F2LLM-0.6B v1 (not v2, whose training data holds xCodeEval itself), pinned to a
# revision. `fetch` downloads the files that match ENCODER_FILE_PATTERNS (huggingface_hub's
# fnmatch patterns, where * also matches "/") into the pretrained cache's folder of the snapshot,
# <cache>/codefuse-ai--F2LLM-0.6B/<revision>/ (codeforces_tags.pretrained.snapshot_folder).
ENCODER_REPO_ID = "codefuse-ai/F2LLM-0.6B"
ENCODER_REVISION = "eb2778d335980b42b986c75d7fe3a78be19ef3a6"
ENCODER_FILE_PATTERNS: tuple[str, ...] = ("*.json", "*.safetensors", "*.txt")
# The files of ENCODER_REPO_ID at ENCODER_REVISION that match ENCODER_FILE_PATTERNS, with
# the sha256 of each (repository path -> sha256): `fetch` checks a download against them
# and a loaded method checks the files it reads. A file stored in git LFS (the weights, the
# tokenizer) has the sha256 the Hub lists; the others were hashed from the files themselves.
ENCODER_FILES: dict[str, str] = {
    "1_Pooling/config.json": "7fd8d0afdbd487713d0991b520b489590739393aa928383f4a853f460b9c2df1",
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "config.json": "a4e75368c654d7f560eedbc19bfc4c4e1951fda51bd99e8c9262c271bc88f15e",
    "config_sentence_transformers.json": (
        "ed6d9b6cef4b09c247757030f0281741ab2671c988e52fcfa311300452bba8b1"
    ),
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "d950e8a1e04cbe3df4acb46a15263c5eb79e3131b8084fb4c3075efc32b3e8fc",
    "modules.json": "21ddb3037a55ebf4549eb5c1dc44c3f28f10b56ec7f1335c92fd90e5b30d88ac",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "38360d5a512a43641b36d6fba2df87b8a3f5464c6b5c76f03e82d6d795175566",
    "tokenizer_config.json": "5509339cb257813101feee0d5d166281482be6a87c464546e1f6d6c027d84d76",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
# A file with one of these suffixes is, or may be, a pickle: `fetch` refuses it anywhere in the
# download, and a method folder never holds one (docs/adr/0004).
PICKLE_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib")

# What is stored for each text: the hidden state of its last token (<|im_end|>, which the
# tokenizer appends), L2-normalised, in float32. 1,024 tokens is the length v1 was trained at.
EOS_TOKEN_ID = 151645
POOLING = "last_token_l2_fp32"
MAX_LENGTH = 1024
# One fixed task instruction per input, prepended to its text to make the encoder input.
STATEMENT_INSTRUCTION = (
    "Instruct: Classify the given competitive programming problem by the algorithmic topics it "
    "involves, such as math, graphs, strings, number theory, trees, geometry, games and "
    "probabilities.\nQuery: "
)
SOLUTION_INSTRUCTION = (
    "Instruct: Classify the given Python solution of a competitive programming problem by the "
    "algorithmic topics it involves, such as math, graphs, strings, number theory, trees, "
    "geometry, games and probabilities.\nQuery: "
)

# Hardware. Threads and batch sizes change how fast the encoder runs, and its output only by
# rounding (the methods' score_atol covers that), so they stay out of SETTINGS and the recipe.
# A batch holds at most *_BATCH_TOKENS padded tokens; on the GPU also at most
# GPU_MAX_BATCH_SEQUENCES sequences: a batch of texts all cut at 1,024 tokens has no padding,
# which makes the attention build its full score matrix (about 4 GiB at 32 sequences).
CPU_THREADS = 6
CPU_BATCH_TOKENS = 8192
GPU_BATCH_TOKENS = 32768
GPU_MAX_BATCH_SEQUENCES = 16
# A loaded method encodes on the device that $CODEFORCES_DEVICE names, "cpu" or "cuda", read
# when it loads (config.DEVICE_ENV_VAR, codeforces_tags.devices.requested_device). Unset, empty,
# or AUTO_DEVICE, it encodes on "cuda" when torch sees a GPU, else on "cpu".
AUTO_DEVICE = "auto"

# The embedding cache, config.ARRAY_CACHE_DIR / CACHE_NAME, shared by the four methods (its keys
# hold the model and the exact text). CHUNK_TEXTS texts are embedded per cache write, so an
# interrupted run loses at most one chunk.
CACHE_NAME = "f2llm"
CHUNK_TEXTS = 256

# Iterations of the lbfgs solver of the logistic regression heads (the linear SVMs use
# config.MODEL_MAX_ITER).
LBFGS_MAX_ITER = 5000

# The grids of the three tuned methods: one dict per cell, in the order the tuning tie rule reads
# (the first of the tied cells wins, so the simpler cells come first: the first head, the
# smallest C, no class weight, the fewest embedding blocks).
_HEADS = ("logistic_regression", "linear_svm", "ridge")
_CLASS_WEIGHTS = ("none", "balanced")
# The probe: head x C x class weight x solution weight, 3 x 7 x 2 x 4 = 168 cells. A solution
# weight of 0 keeps the statement embedding alone.
PROBE_GRID: tuple[dict, ...] = tuple(
    {"head": head, "c": c, "class_weight": class_weight, "solution_weight": solution_weight}
    for head, c, class_weight, solution_weight in itertools.product(
        _HEADS,
        (0.01, 0.0464, 0.215, 1.0, 4.64, 21.5, 100.0),
        _CLASS_WEIGHTS,
        (0.0, 0.25, 0.5, 1.0),
    )
)
# Early fusion: C x statement-embedding weight x solution-embedding weight, 5 x 4 x 4 = 80 cells.
EARLY_FUSION_GRID: tuple[dict, ...] = tuple(
    {"c": c, "statement_weight": statement_weight, "solution_weight": solution_weight}
    for c, statement_weight, solution_weight in itertools.product(
        (0.2, 0.4, 0.62, 1.0, 1.6),
        (0.1, 0.25, 0.5, 1.0),
        (0.0, 0.1, 0.25, 0.5),
    )
)
# Late fusion: the weight of the TF-IDF scores, 0.05 to 0.95 in steps of 0.05 (19 cells). 0 and 1
# are the probe and TF-IDF alone, compared on their own.
LATE_FUSION_GRID: tuple[dict, ...] = tuple(
    {"weight": round(0.05 * step, 2)} for step in range(1, 20)
)
GRIDS: dict[str, tuple[dict, ...]] = {
    PROBE: PROBE_GRID,
    LATE_FUSION: LATE_FUSION_GRID,
    EARLY_FUSION: EARLY_FUSION_GRID,
}
# Late fusion standardises each model's scores with those of this many inner group folds.
FUSION_INNER_FOLDS = 5

# SetFit, fixed before any result (nothing is chosen on the out-of-fold labels): 375 positive
# and 375 negative pairs per focus tag, one epoch of 16-pair steps built from 8-pair
# micro-batches, LoRA on every linear layer of the encoder. SETFIT_SMOKE_PAIRS pairs train the
# short run that measures score_atol.
SETFIT: dict = {
    "pairs_per_tag_positive": 375,
    "batch_pairs": 16,
    "micro_batch_pairs": 8,
    "learning_rate": 1e-4,
    "epochs": 1,
    "warmup": 0.1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "seed": 0,
}
SETFIT_SMOKE_PAIRS = 100
LORA_TARGETS: tuple[str, ...] = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)

# The files the four methods share, in assets/f2llm/: not under assets/methods/, where CI checks
# every folder as a frozen method.
SHARED_ASSETS_DIR = config.PROJECT_ROOT / "assets" / "f2llm"
# embedding_pairs.csv flags the pairs of problems of different near-duplicate groups whose
# statement embeddings have a cosine at or above this quantile of all such pairs' cosines.
PAIRS_QUANTILE = 0.9999
# The frozen TF-IDF model's macro F1 on the test set (notebook 04 prints it rounded to 0.689),
# the project's config.TFIDF_TEST_MACRO_F1, which tuning freeze also checks: tfidf_reference.csv
# is written only if the rows it holds reproduce it.
TFIDF_TEST_MACRO_F1 = config.TFIDF_TEST_MACRO_F1


def experiments_dir(name: str) -> Path:
    """Return the local folder of a method under ``experiments/``.

    It holds the Optuna journal and the tuning results of a tuned method, and SetFit's saved
    adapters: gitignored files whose committed record is ``assets/methods/<name>/``.

    Args:
        name: The method's registry key, one of ``NAMES``.

    Returns:
        ``config.EXPERIMENTS_DIR / name``, read when called.
    """
    return config.EXPERIMENTS_DIR / name


def _grid_summary(grid: tuple[dict, ...]) -> dict[str, object]:
    """Summarise a grid by its size and the sha256 of its cells, as the recipe holds it."""
    return {"size": len(grid), "sha256": sha256_json(list(grid))}


# What a method's recipe_sha256 and run record hold under "settings": JSON values only, and short
# enough to stay under MLflow's limit on the length of a parameter once the run record is
# flattened (a grid is summarised; its cells live in full above).
SETTINGS: dict = {
    "encoder_repo_id": ENCODER_REPO_ID,
    "encoder_revision": ENCODER_REVISION,
    "encoder_files": dict(ENCODER_FILES),
    "eos_token_id": EOS_TOKEN_ID,
    "pooling": POOLING,
    "max_length": MAX_LENGTH,
    "statement_instruction": STATEMENT_INSTRUCTION,
    "solution_instruction": SOLUTION_INSTRUCTION,
    "lbfgs_max_iter": LBFGS_MAX_ITER,
    "probe_grid": _grid_summary(PROBE_GRID),
    "late_fusion_grid": _grid_summary(LATE_FUSION_GRID),
    "early_fusion_grid": _grid_summary(EARLY_FUSION_GRID),
    "fusion_inner_folds": FUSION_INNER_FOLDS,
    "setfit": dict(SETFIT),
    "lora_targets": list(LORA_TARGETS),
}

# score_atol, the tolerance of verify on the four methods' scores, is this many times
# the largest gap measured between the fit-time scores (cached GPU embeddings, SetFit's body
# merged on the GPU) and a loaded method's on the CPU (live embeddings, one problem at a time).
SCORE_ATOL_FACTOR = 10
