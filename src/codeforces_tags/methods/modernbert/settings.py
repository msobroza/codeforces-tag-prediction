"""Settings of the two ModernBERT methods: what their recipes hash (docs/adr/0004, 0005).

Every value is a JSON scalar, tuple or dict, so a variant's ``asdict`` goes into the run record's
``method_entries["settings"]`` and into the recipe as it is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# The registry keys of the two methods.
LORA = "modernbert_lora"
FULL = "modernbert_base_full"

# The files of a method folder (beside method.json) and of a fold model in the trained cache.
BAG_FILE = "bag.json"
PRETRAINED_FILE = "pretrained.json"
TRAINED_FILE = "trained.json"
FOLD_WEIGHTS_FILE = "weights.safetensors"
TRAINING_FILE = "training.json"
HELD_OUT_FILE = "held_out_scores.npy"
# The extra frozen tables of a method, beside the framework's frozen files: each fold model's
# test scores, and the CPU timings (a latency.csv, in the folder analysis/cpu/).
ANALYSIS_FOLDER = "analysis"
FOLD_TEST_SCORES_FILE = "fold_test_scores.csv"
CPU_LATENCY_FOLDER = "cpu"
# The study's journal, in experiments/<name>/, and the trained cache's folder of the fold models
# that verify --retrain trains again, beside the frozen ones.
JOURNAL_FILE = "optuna.jsonl"
RETRAIN_FOLDER = "retrain"
# The frozen results of the tuning study (CLAUDE.md, Adding a tagging method, step 1).
TUNING_FOLDER = "tuning"
PARAMS_FILE = "params.json"
TRIALS_FILE = "trials.csv"
HISTORY_FILE = "history.csv"
IMPORTANCES_FILE = "importances.csv"
OUT_OF_FOLD_FILE = "out_of_fold.csv"

# The files of a pretrained snapshot that predicting needs: the configuration and the tokenizer,
# and the encoder weights when the fold models hold only what LoRA trained.
CONFIG_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ENCODER_FILE = "model.safetensors"


@dataclass(frozen=True)
class Variant:  # pylint: disable=too-many-instance-attributes
    """The settings of one method.

    Attributes:
        name: Its registry key.
        study_name: The name of its Optuna study in ``experiments/<name>/optuna.jsonl``.
        model_repo: The pretrained encoder on the Hugging Face Hub.
        model_revision: The commit of ``model_repo`` its weights come from.
        lora: True when only LoRA updates, the pooling head and the classifier train; False when
            every weight trains.
        max_length: Tokens per input, ``[CLS] statement [SEP] solution [SEP]``.
        search_space: The study's ranges, by parameter: ``(low, high)`` or a tuple of choices.
        lora_targets: The linear layers that get a LoRA update (LoRA only).
        trained_modules: The modules trained in full beside the updates (LoRA only).
        lora_alpha_per_rank: LoRA's scale alpha divided by its rank (LoRA only).
        weight_decay: AdamW's weight decay when the study does not tune it (LoRA only).
        code_min_tokens: Solution tokens kept whatever the statement's length.
        micro_batch: Problems per forward pass in training; gradient accumulation does the rest.
        score_batch: Problems per forward pass when scoring.
        pad_to_multiple_of: Batches are padded to a multiple of this many tokens.
        seed: The seed of every training (the first one tried when a fold model collapses).
        tuning_fold: The dev fold the study scores; it trains on the others.
        subfolds: The stratified group sub-folds of the tuning fold that cross-fit its thresholds.
        tuning_trials: The finished trials (complete or pruned) the study runs to.
        startup_trials: Random trials before TPE takes over.
        pruner_min_peers: Finished trials of the same epoch count needed before pruning.
        max_failed_trials: Failed trials (out of memory, diverged) allowed beside the finished ones.
        collapse_ap: A fold model whose macro average precision on its own training sample stays
            under this is trained again with the next seed.
        collapse_sample: The training problems that measure it.
        fold_seeds: The seeds tried per fold before giving up.
        device: ``"auto"`` predicts on CUDA when it is available, else on the CPU.
        retrain_min_correlation: ``verify --retrain`` passes when, for every fold model and tag,
            the Pearson correlation of its retrained and frozen out-of-fold probabilities
            reaches this,
        retrain_max_median_gap: when the median gap between those probabilities is at most
            this,
        retrain_max_flips: when at most this share of the dev 0/1 predictions flips,
        retrain_f1_margin: and when the interval of the paired difference of the dev macro F1
            lies within plus or minus this. Declared before any retraining: two different
            models of the same macro F1 (modernbert_lora's and modernbert_base_full's
            out-of-fold scores: correlations 0.67 at worst, median gap 0.045, 4.9% flips) fail
            the first three.
    """

    name: str
    study_name: str
    model_repo: str
    model_revision: str
    lora: bool
    max_length: int
    search_space: dict = field(default_factory=dict)
    lora_targets: tuple[str, ...] = ()
    trained_modules: tuple[str, ...] = ()
    lora_alpha_per_rank: int = 0
    weight_decay: float = 0.0
    code_min_tokens: int = 128
    micro_batch: int = 4
    score_batch: int = 16
    pad_to_multiple_of: int = 8
    seed: int = 0
    tuning_fold: int = 0
    subfolds: int = 5
    tuning_trials: int = 30
    startup_trials: int = 10
    pruner_min_peers: int = 3
    max_failed_trials: int = 5
    collapse_ap: float = 0.3
    collapse_sample: int = 800
    fold_seeds: int = 3
    device: str = "auto"
    retrain_min_correlation: float = 0.9
    retrain_max_median_gap: float = 0.02
    retrain_max_flips: float = 0.025
    retrain_f1_margin: float = 0.03

    @property
    def pretrained_files(self) -> tuple[str, ...]:
        """The files of the pretrained snapshot that predicting reads, pinned in pretrained.json.

        Full fine-tuning's fold models hold every weight, so they need no encoder weights.
        """
        return (*CONFIG_FILES, ENCODER_FILE) if self.lora else CONFIG_FILES


VARIANTS = {
    LORA: Variant(
        name=LORA,
        study_name="modernbert-lora",
        model_repo="answerdotai/ModernBERT-large",
        model_revision="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        lora=True,
        max_length=1024,
        search_space={
            "learning_rate": (5e-5, 1e-3),  # log scale
            "lora_rank": (8, 16, 32),
            "lora_dropout": (0.0, 0.2),
            "epochs": (2, 4),
            "batch_size": (16, 32),
            "warmup_ratio": (0.0, 0.2),
            "pos_weight_power": (0.0, 1.0),
        },
        lora_targets=("Wqkv", "Wo", "Wi"),
        trained_modules=("head", "classifier"),
        lora_alpha_per_rank=2,
        weight_decay=0.01,
    ),
    FULL: Variant(
        name=FULL,
        study_name="modernbert-base-full",
        model_repo="answerdotai/ModernBERT-base",
        model_revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        lora=False,
        max_length=2048,
        search_space={
            "learning_rate": (1e-5, 2e-4),  # log scale
            "weight_decay": (1e-5, 1e-1),  # log scale
            "classifier_dropout": (0.0, 0.2),
            "epochs": (2, 5),
            "batch_size": (16, 32),
            "warmup_ratio": (0.0, 0.2),
            "pos_weight_power": (0.0, 1.0),
        },
    ),
}


def settings(variant: Variant) -> dict:
    """Return a method's settings as plain JSON values, for its run record and recipe.

    Args:
        variant: The method's settings.

    Returns:
        Every field, tuples as lists.
    """
    return _json_values(asdict(variant))


def study_recipe(variant: Variant) -> dict:
    """Return the recipe its Optuna study recorded, which a study refuses to resume under another.

    These are the keys and values that the former notebooks 05 and 06 (deleted; their runs are
    in experiments/<name>/) stored in their journals, so the package resumes and exports those
    studies.

    Args:
        variant: The method's settings.

    Returns:
        The recipe, with plain JSON values.
    """
    recipe = {
        "model_repo": variant.model_repo,
        "model_revision": variant.model_revision,
        "max_length": variant.max_length,
        "code_min_tokens": variant.code_min_tokens,
        "micro_batch": variant.micro_batch,
    }
    if variant.lora:
        recipe |= {
            "lora_targets": variant.lora_targets,
            "trained_modules": variant.trained_modules,
            "lora_alpha_per_rank": variant.lora_alpha_per_rank,
            "weight_decay": variant.weight_decay,
        }
    recipe |= {
        "seed": variant.seed,
        "tuning_fold": variant.tuning_fold,
        "subfolds": variant.subfolds,
        "search_space": variant.search_space,
        "smoke": False,
    }
    return _json_values(recipe)


def _json_values(value):
    """Turn tuples into lists, recursively, as a JSON round trip would."""
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value
