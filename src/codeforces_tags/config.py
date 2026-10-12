"""Project settings, gathered in one file.

Every value that controls the package's behaviour lives here: where the data is, which tags
the task focuses on, and the defaults of the analysis helpers. Tool settings (ruff,
pylint, pytest, coverage) live in ``pyproject.toml``.
"""

import zipfile
from pathlib import Path

# Data ----------------------------------------------------------------------------------------

# Repository root under the editable install made by `uv sync`. When the package is installed
# elsewhere (e.g. as a wheel), pass data_dir explicitly or set $CODEFORCES_DATA_DIR, and pass
# verdicts=load_verdicts(path) to near_duplicate_groups().
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_ENV_VAR = "CODEFORCES_DATA_DIR"
DEFAULT_DATA_DIR = PROJECT_ROOT / "code_classification_dataset"
# The dataset every frozen result was computed on: check_dataset() refuses any other. The sha256
# runs over the (file name, NUL, bytes, NUL) of each sample_<id>.json file, in file name order
# (provenance.dataset_sha256), computed on the 4,982 files of the challenge dataset.
DATASET_FILES = 4982
DATASET_SHA256 = "266d2792ebbc985570cf804a540cecc6248c6ed44b1bf298cb13bf38b937e413"

# Labels --------------------------------------------------------------------------------------

# The 8 focus tags, in the order of the task statement (PROBLEM.md).
FOCUS_TAGS: tuple[str, ...] = (
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
)
# Every tag of the dataset, in sorted() order: the 37 tags a method predicting all tags encodes
# (multi_hot(..., vocabulary=ALL_TAGS)), because multi_hot infers its vocabulary from its data
# and the test set alone has 36 (no "schedules").
ALL_TAGS: tuple[str, ...] = (
    "*special",
    "2-sat",
    "binary search",
    "bitmasks",
    "brute force",
    "chinese remainder theorem",
    "combinatorics",
    "constructive algorithms",
    "data structures",
    "dfs and similar",
    "divide and conquer",
    "dp",
    "dsu",
    "expression parsing",
    "fft",
    "flows",
    "games",
    "geometry",
    "graph matchings",
    "graphs",
    "greedy",
    "hashing",
    "implementation",
    "interactive",
    "math",
    "matrices",
    "meet-in-the-middle",
    "number theory",
    "probabilities",
    "schedules",
    "shortest paths",
    "sortings",
    "string suffix structures",
    "strings",
    "ternary search",
    "trees",
    "two pointers",
)
# The 31 tags of the xCodeEval paper's tag classification results (its Fig. 8), in ALL_TAGS
# order: ALL_TAGS without "*special", "chinese remainder theorem", "hashing", "schedules",
# "string suffix structures" and "ternary search".
PAPER_TAGS: tuple[str, ...] = (
    "2-sat",
    "binary search",
    "bitmasks",
    "brute force",
    "combinatorics",
    "constructive algorithms",
    "data structures",
    "dfs and similar",
    "divide and conquer",
    "dp",
    "dsu",
    "expression parsing",
    "fft",
    "flows",
    "games",
    "geometry",
    "graph matchings",
    "graphs",
    "greedy",
    "implementation",
    "interactive",
    "math",
    "matrices",
    "meet-in-the-middle",
    "number theory",
    "probabilities",
    "shortest paths",
    "sortings",
    "strings",
    "trees",
    "two pointers",
)

# Preprocessing -------------------------------------------------------------------------------

# Free-text fields cleaned by `preprocess()` (glued paragraphs, look-alike letters).
TEXT_FIELDS: tuple[str, ...] = (
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
    "prob_desc_notes",
)
# Fields joined into the `statement` column, the problem text a model reads.
STATEMENT_FIELDS: tuple[str, ...] = (
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
)
# A text is treated as Russian when more than this share of its letters is Cyrillic. Below it,
# Cyrillic letters are look-alikes typed in English text ("С++" with a Cyrillic "С").
RUSSIAN_LETTER_SHARE = 0.5
# The raw columns every tagging method reads, as read_samples() gives them: raw_to_problems()
# refuses raw problems without one, because preprocess() silently skips a missing column.
RAW_INPUT_COLUMNS: tuple[str, ...] = (
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
    "prob_desc_notes",
    "prob_desc_sample_inputs",
    "prob_desc_sample_outputs",
    "source_code",
)
# The columns a tagging method never reads: the labels, and the split's columns that
# select_part() adds. MethodSpec refuses them in its input_columns, which check_tagging_method
# restricts a method's input to.
LABEL_AND_SPLIT_COLUMNS: tuple[str, ...] = ("tags", "part", "fold", "group")

# Near-duplicate statements -------------------------------------------------------------------

# Detection has two stages, chosen by the audit in notebooks/02_near_duplicates.ipynb.
# Field compared: the problem description alone, which the audit's threshold and verdicts used.
NEAR_DUPLICATE_FIELD = "prob_desc_description"
# Stage 1: TF-IDF cosine similarity at which two descriptions become candidate duplicates. It
# is low for recall: every duplicate the audit confirmed scores at least 0.497.
NEAR_DUPLICATE_CANDIDATE_THRESHOLD = 0.4
# Stage 2: the frozen verdicts of two blind LLM judges on the candidate pairs, the results of
# the audit (its prompt templates live in the package, see AUDIT_TEMPLATE_DIR).
NEAR_DUPLICATE_AUDIT_DIR = PROJECT_ROOT / "assets" / "near_duplicates"
NEAR_DUPLICATE_VERDICTS = NEAR_DUPLICATE_AUDIT_DIR / "judgments.csv"

# Near-duplicate audit (python -m codeforces_tags.near_duplicates.audit) ----------------------

# The values of the recorded run (2026-10-04). Changing one changes the pairs or the verdicts.
# Candidate signals besides the candidate threshold: a pair is flagged when any of them fires.
AUDIT_SAME_ROUND_WORD_THRESHOLD = 0.25  # word similarity, for two problems of the same round
AUDIT_CHAR_THRESHOLD = 0.6  # character n-gram similarity of the descriptions
AUDIT_CHAR_NGRAMS = (3, 5)  # character n-gram sizes, within word boundaries
AUDIT_CODE_THRESHOLD = 0.85  # token similarity of the accepted solutions
AUDIT_CODE_TOKENS = r"(?u)\b\w+\b"  # code tokens: every identifier, keyword and number
AUDIT_VERSION_PATTERN = r"\b(?:easy|hard|easier|harder)\s+version"  # announces a version
AUDIT_SIMILARITY_DECIMALS = 3  # similarities are saved rounded
# Tiers C and D are random samples of this size, drawn with these seeds.
AUDIT_WEAK_SIGNAL_SAMPLE = 60
AUDIT_CONTROL_SAMPLE = 150
AUDIT_SEEDS = {"weak_signal_sample": 0, "control_sample": 1, "shuffle": 2}
# What a judge reads: the first characters of each side's description and specs.
AUDIT_TEXT_LIMITS = {"description": 2500, "input_spec": 700, "output_spec": 400}
# Pairs per judge, at most: the batches grow in number, not in size, with the data (the
# recorded run: 485 pairs in 10 batches, then 463 in 9).
AUDIT_PAIRS_PER_JUDGE = 52
# The detector before the audit (TF-IDF only). Round 1 of the second pass re-judges the calls
# that decide its errors.
AUDIT_FORMER_THRESHOLD = 0.8
# The judges: Jinja2 prompt templates and answer schema, model, reasoning effort and tools.
# The recorded run used Claude Code 2.1.284 workflow subagents with this model and effort,
# harness defaults for everything else (temperature, token limits), and structured output.
AUDIT_TEMPLATE_DIR = Path(__file__).parent / "near_duplicates" / "audit" / "templates"
AUDIT_JUDGE_MODEL = "claude-opus-5-5"
AUDIT_JUDGE_EFFORT = "xhigh"
AUDIT_JUDGE_TOOLS = ("Read",)  # read-only: a judge only needs to read its batch file

# Train/test split ----------------------------------------------------------------------------

# Whole near-duplicate groups go to 5 folds by iterative stratification, one stratum at a time
# (groups with a focus tag, groups without). Fold 0 is the test set (20%); the same rule splits
# the rest (dev) into the 5 cross-validation folds.
SPLIT_FOLDS = 5
SPLIT_TEST_FOLD = 0
SPLIT_SEED = 0
# Pairs of focus tags shared by at least this many problems (about 4 per fold) are labels of the
# focus stratum too, so that how tags co-occur is balanced across folds. Rarer pairs have too few
# problems to balance. Any value from 16 to 23 picks the same 9 pairs: the 9th has 23 problems,
# the 10th 15 (notebook 03). Pairs are counted on the problems being split, so the dev folds
# balance the pairs frequent in dev.
SPLIT_MIN_PAIR_PROBLEMS = 20
# Seeds drawn per method when notebook 03 compares the split with random ones.
SPLIT_COMPARISON_SEEDS = 50
# The frozen split: one row per problem, with its part (dev or test), its dev fold and its
# near-duplicate group.
SPLIT_FILE = PROJECT_ROOT / "assets" / "split" / "split.csv"

# Distinctive terms ---------------------------------------------------------------------------

TERMS_TOP_K = 10  # terms reported per label
TERMS_MIN_DF = 10  # ignore terms found in fewer documents
TERMS_MIN_ODDS_RATIO = 2.0  # ignore terms that are only weakly associated with the label

# Plots ---------------------------------------------------------------------------------------

FOCUS_COLOR = "#1f77b4"
OTHER_COLOR = "#c7c7c7"
FIGURE_SIZE = (10, 4.5)
FIGURE_DPI = 100

# Model (codeforces_tags.model) ---------------------------------------------------------------

MODEL_SEED = 0  # random_state of liblinear and LinearSVC
MODEL_MAX_ITER = 5000  # LinearSVC iterations: headroom for the largest C of the search space
# Sample I/O cues: the tree edge list needs at least this many nodes (n = 3 matches multi-test
# pairs such as "3 4 / 3 2 / 3 1"); the player words of games' sample outputs.
CUE_TREE_MIN_NODES = 4
# Lines of one sample input the edge-list cues read: the dataset's longest has 101, and the
# search for an edge list is quadratic in the lines read.
CUE_MAX_SAMPLE_LINES = 200
CUE_PLAYER_WORDS = (
    "alice",
    "bob",
    "first",
    "second",
    "draw",
    "petya",
    "vasya",
    "win",
    "lose",
    "winner",
    "loser",
)

# Evaluation (codeforces_tags.evaluation) -----------------------------------------------------

RELEASE_ERA_FIRST_YEAR = 2018  # statements switched to $$$ LaTeX in 2018
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0
# Out-of-fold ablations of notebook 04: name -> feature blocks kept.
ABLATIONS = {
    "all blocks": ("statement", "code", "cues"),
    "statement + code": ("statement", "code"),
    "statement": ("statement",),
    "code": ("code",),
    "cues": ("cues",),
}
ERROR_EXCERPT_CHARS = 300  # statement characters shown per misclassified problem

# Calibration (codeforces_tags.evaluation.calibration) ----------------------------------------

# Notebook 04 section 5 and `tracking calibrate` ask whether the out-of-fold scores read as
# probabilities. They change no prediction, so none of these settings is in RECORDED_SETTINGS:
# one there would change the tfidf recipe.
# Bins of a reliability diagram, and of the histograms of predicted probabilities under it; also
# the equal-width bins of a reliability table (analysis.calibration), which compares methods bin
# by bin.
CALIBRATION_BINS = 10
# CalibrationDisplay's binning: "quantile" bins hold equal counts. Uniform bins would leave the
# high end of a rare tag nearly empty (74 dev positives for probabilities).
CALIBRATION_STRATEGY = "quantile"
# The probabilities compared, id -> name: the first look's sigmoid, then the tfidf model's SVM
# margins min-max scaled, Platt-scaled or isotonic-calibrated on the other folds. The ids key the
# table and the MLflow metrics; the names label the figures.
CALIBRATION_METHODS: dict[str, str] = {
    "logistic_regression": "logistic regression",
    "svm_minmax": "SVM, min-max",
    "svm_platt": "SVM + Platt",
    "svm_isotonic": "SVM + isotonic",
}

# Threshold rule (codeforces_tags.model.platt_lipton) -----------------------------------------

# The Platt + Lipton threshold rule of the calibrated Jev twins (docs/adr/0011); notebook 11's
# helpers (analysis.threshold_rules, analysis.calibrators) read them too. A twin's recipe holds
# them through platt_lipton.RULE_SETTINGS, so none is in RECORDED_SETTINGS: one there would change
# every frozen method's recipe.
# The scores are clipped to [PLATT_CLIP, 1 - PLATT_CLIP] before their logit: only a failed
# answer's 0.0 reaches the clip, and its logit stays finite.
PLATT_CLIP = 1e-4
# The step of Jev's probabilities, 0.01 to 0.99 (a failed answer scores 0.0): the rule snaps a
# threshold to the midpoint below a grid value.
SCORE_GRID = 0.01
# The rule's Platt map is the maximum-likelihood fit, so that a threshold does not depend on when
# a solver stopped: lbfgs at scikit-learn's default tolerance (1e-4) stops a few iterations in,
# far enough from the maximum to move a snapped threshold by one grid step. Newton's method, on
# the Cholesky factor of the Hessian of one or two features, reaches PLATT_TOL in a few
# iterations, well within scikit-learn's default max_iter (100).
PLATT_SOLVER = "newton-cholesky"
PLATT_TOL = 1e-12

# Tuning (python -m codeforces_tags.tuning) ---------------------------------------------------

TUNING_STUDY = "tfidf-baseline"
TUNING_TRIALS = 200
TUNING_WORKERS = 4
TUNING_SEED = 0  # worker i samples with TUNING_SEED + i
TUNING_STARTUP_TRIALS = 30
# BLAS threads per worker: liblinear's BLAS calls would otherwise start one thread per CPU in
# each worker (4 workers on 12 CPUs ran twice slower, with the same predictions).
TUNING_BLAS_THREADS = 1
# export saves nothing when its re-run of the best configuration scores further than this from
# the study's best value: the code, data or split then differ from the study's. (The BLAS
# thread count changes the scores by rounding errors only: tests/model/test_training.py.)
TUNING_EXPORT_TOLERANCE = 1e-9
# The test macro F1 of the tuned tag model, as notebook 04 scored it before the tag model became
# the tagging method tfidf: tuning freeze writes nothing, and verify fails, unless the test
# predictions score exactly this. It pins the frozen configuration: another one is a new model
# (docs/adr/0004), which tuning freeze refuses without printing its test score.
TFIDF_TEST_MACRO_F1 = 0.6887545239153905
TUNING_SPACE = {
    "c_logistic_regression": (0.01, 30.0),
    "c_linear_svm": (0.001, 3.0),
    "statement_ngram_max": (1, 3),
    "statement_min_df": (1, 5),
    "statement_max_df": (0.5, 1.0),
    "code_weight": (0.1, 1.5),
    "code_ngram_max": (1, 3),
    "code_min_df": (2, 10),
    "cue_weight": (0.05, 1.0),
}
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"  # gitignored: Optuna journal, MLflow store
OPTUNA_JOURNAL = EXPERIMENTS_DIR / "optuna_journal.jsonl"
MLFLOW_TRACKING_URI = f"sqlite:///{EXPERIMENTS_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = EXPERIMENTS_DIR / "mlartifacts"
MLFLOW_EXPERIMENT = "tfidf-baseline"
TFIDF_BASELINE_DIR = PROJECT_ROOT / "assets" / "tfidf_baseline"

# Tagging methods (codeforces_tags.methods) ---------------------------------------------------

# The format of method.json, the file that names a method folder's method and tags; load_method
# refuses any other version.
METHOD_FORMAT_VERSION = 1
# Each frozen method's files (predictions, scores, run record, method folder) live in
# METHODS_ASSETS_DIR / <name>.
METHODS_ASSETS_DIR = PROJECT_ROOT / "assets" / "methods"
# Read for the dependency groups a method needs (installed_methods) and the pinned versions of
# its requirements.
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_FILE = PROJECT_ROOT / "uv.lock"

# Provenance (codeforces_tags.provenance) -----------------------------------------------------

# The packages whose installed versions every run record holds, besides a method's own
# requirements.
RECORDED_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "scipy", "skops", "mlflow")
# The project-wide settings every run record holds (settings_snapshot()). A method's own
# settings live in its package's settings.py, and its run record holds them in
# method_entries["settings"].
RECORDED_SETTINGS: tuple[str, ...] = (
    "MODEL_SEED",
    "MODEL_MAX_ITER",
    "CUE_TREE_MIN_NODES",
    "CUE_MAX_SAMPLE_LINES",
    "CUE_PLAYER_WORDS",
    "TEXT_FIELDS",
    "STATEMENT_FIELDS",
    "RUSSIAN_LETTER_SHARE",
    "FOCUS_TAGS",
)

# Array cache (codeforces_tags.cache) ---------------------------------------------------------

# Costly arrays (e.g. an encoder's embeddings), one .npy file per key, in ARRAY_CACHE_DIR /
# <method name>. A fit-time option only: a loaded method has no cache.
ARRAY_CACHE_DIR = EXPERIMENTS_DIR / "cache"

# Self-identifying solutions (python -m codeforces_tags.evaluation slices) --------------------

# The contamination-sensitivity slice: the problems whose solution names or quotes its own
# problem. Its rule was fixed before any API call; the slices command freezes it once in
# SELF_IDENTIFYING_FILE, and notebooks report every score also without these problems.
SLICES_DIR = PROJECT_ROOT / "assets" / "slices"
SELF_IDENTIFYING_FILE = SLICES_DIR / "self_identifying_solutions.csv"
# Rule name -> regex searched in the solution: a Codeforces contest/problemset/gym URL, a
# CodeforcesTask<contest><letter> class, a "Codeforces [Round] #N" or "Codeforces <id>" header,
# a "cf NNN X" problem id.
SELF_IDENTIFYING_PATTERNS: dict[str, str] = {
    "codeforces_url": r"(?i)codeforces\.(?:com|ru)/(?:contest|problemset|gym)/",
    "task_class": r"CodeforcesTask\d+[A-Z]",
    "round_or_id_header": r"(?i)codeforces(?:\.com|\.ru)?\s*(?:round)?\s*#?\s*\d+",
    "cf_problem_id": r"(?i)\bcf\s*#?\d{2,4}\s*[A-H]\b",
}
# The last rule, quotes_description: the solution's comments and triple-quoted strings hold at
# least this share of the description's distinct word n-grams, of this size.
SELF_IDENTIFYING_NGRAM = 5
SELF_IDENTIFYING_MIN_SHARE = 0.5

# Latency (codeforces_tags.evaluation.timing) -------------------------------------------------

# PROBLEM.md: a single-problem prediction must finish in under ten seconds. plot_latency draws
# this budget, and ADR 0004 makes a frozen method eligible for the CLI only when the slowest
# problem of its latency.csv is under it.
LATENCY_BUDGET_S = 10.0
# `tracking latency` times this many dev problems per method, drawn with this seed (pandas'
# DataFrame.sample), one at a time and once in a single call, and writes them once to the
# method's latency.csv.
LATENCY_PROBLEMS = 50
LATENCY_SEED = 0

# Command line (codeforces_tags.cli) ----------------------------------------------------------

# The method that `codeforces-tags predict` and `evaluate` run without --method: the user's choice
# (docs/adr/0008).
CLI_DEFAULT_METHOD = "jev-statement-code"
# The method every failure of the command line names: local, offline and free (no dependency
# group, no API). The command line never falls back on it, or on any other method, by itself.
CLI_OFFLINE_METHOD = "tfidf"
# Dependency group -> the API key its SDK reads. Before it loads a method of the group that calls
# an API, the command line checks that the key is set and not empty; it never reads it further,
# and never prints it.
API_KEY_ENV_VARS: dict[str, str] = {
    "llm": "OPENAI_API_KEY",
    "llm-typesafe": "TYPESAFE_API_KEY",
}
# Dependency group -> the base class of its SDK's errors. The command line looks it up in
# sys.modules once an error is raised, never imports it, and reports such an error (a refused key,
# no credit) by its class and status, never by its text, which can echo the state.
API_SDK_ERRORS: dict[str, str] = {
    "llm": "openai.OpenAIError",
    "llm-typesafe": "typesafe_sdk.TypeSafeError",
}
# The fields a sample must hold as a text (a string with something besides whitespace), as Jev's
# requests need them. Any other raw field may be missing from a sample file: it reads as null, as
# the dataset's null fields do.
CLI_REQUIRED_FIELDS: tuple[str, ...] = ("prob_desc_description", "source_code")
# The raw fields that hold a list of texts, which the dataset writes as its JSON text and
# clean_columns decodes. A sample file may also hold such a field as a list: the command line
# writes it as the dataset does. Every other raw field of a sample holds a text or null.
CLI_JSON_LIST_FIELDS: tuple[str, ...] = ("prob_desc_sample_inputs", "prob_desc_sample_outputs")
# A method that calls an API is asked this many problems at a time (the TypeSafe methods'
# CONCURRENCY). The cost cap is checked between batches, so a run overshoots it by one batch at
# most (about $0.002 for Jev).
CLI_API_BATCH = 16
# The most paid requests sent without --max-cost-usd: one prediction (about $0.0001 for Jev).
CLI_UNCAPPED_REQUESTS = 1
# predict prints a request's cost_usd rounded to this many decimals: it drops the float noise of a
# sum of prices (0.00012600000000000003) and keeps a token of every model of LLM_PRICES.
CLI_COST_DECIMALS = 10

# Method folders (codeforces_tags.conformance) ------------------------------------------------

# A method folder holds data files only, at most METHOD_FILE_MAX_BYTES each and
# METHOD_FOLDER_MAX_BYTES in all: check_method_folder refuses a larger one, so it stays small
# enough to commit. Pretrained weights never go in it (docs/adr/0004).
METHOD_FILE_MAX_BYTES = 50 * 1024**2
METHOD_FOLDER_MAX_BYTES = 100 * 1024**2
# Pretrained weights, downloaded once per machine by a method's fetch command, under the
# gitignored experiments/; $CODEFORCES_PRETRAINED_DIR, read when a method loads (as read_samples
# reads $CODEFORCES_DATA_DIR), overrides it when it is not empty. One folder per pinned snapshot,
# <cache>/<org>--<name>/<revision> for every method: codeforces_tags.pretrained resolves both
# (pretrained_root, snapshot_folder).
PRETRAINED_CACHE_DIR = EXPERIMENTS_DIR / "pretrained"
PRETRAINED_DIR_ENV_VAR = "CODEFORCES_PRETRAINED_DIR"
# Trained weights too large for a method folder (the fine-tuned fold models of
# methods/modernbert/), written by the method's train or adopt command under the gitignored
# experiments/ and named, with their sha256, by the folder's trained.json; $CODEFORCES_TRAINED_DIR,
# read when a method loads, overrides it when it is not empty (docs/adr/0006).
TRAINED_CACHE_DIR = EXPERIMENTS_DIR / "trained"
TRAINED_DIR_ENV_VAR = "CODEFORCES_TRAINED_DIR"
# The compression of the skops files a method saves (skops.io.dump(..., compression=...)).
SKOPS_COMPRESSION = zipfile.ZIP_DEFLATED

# Devices (codeforces_tags.devices) -----------------------------------------------------------

# The device a method that runs torch runs on, chosen where it is deployed, not in its settings:
# $CODEFORCES_DEVICE, read at call time by devices.requested_device, names one of DEVICES
# ("auto": CUDA when torch sees a GPU, else the CPU); unset or empty, the caller's default.
# LEGACY_DEVICE_ENV_VAR, the variable the F2LLM methods read before, is refused whenever it is
# set, naming DEVICE_ENV_VAR: a device variable silently ignored, on a machine whose GPU once
# rebooted it, is the failure to avoid. None of them is in RECORDED_SETTINGS: a run record
# holds the device a method ran on, never the variable that chose it.
DEVICE_ENV_VAR = "CODEFORCES_DEVICE"
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
LEGACY_DEVICE_ENV_VAR = "CODEFORCES_F2LLM_DEVICE"

# Tagging methods in MLflow (codeforces_tags.tracking) ----------------------------------------

# The experiment of every method run, evaluation dataset and evaluation run, in the store of
# MLFLOW_TRACKING_URI (MLFLOW_EXPERIMENT keeps the tuning study's runs).
MLFLOW_METHODS_EXPERIMENT = "tag-methods"
# The environment tracking.environment.mlflow_env() sets, the only code that sets MLflow's
# environment (docs/adr/0002, safeguard 7), whatever the environment held before.
MLFLOW_ENV: dict[str, str] = {
    # Never load a pickle: loading one runs any code it holds.
    "MLFLOW_ALLOW_PICKLE_DESERIALIZATION": "false",
    # Explicit requirements: otherwise each logged model copies uv.lock (990 KB) and lists 173
    # requirements.
    "MLFLOW_UV_AUTO_DETECT": "false",
    "MLFLOW_LOG_UV_FILES": "false",
    # One evaluation worker: the default 10 exhaust sqlite's connection pool and silently drop
    # evaluation rows.
    "MLFLOW_GENAI_EVAL_MAX_WORKERS": "1",
    # Synchronous trace export: the asynchronous queue silently drops traces.
    "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
    # Never fetch MLflow's model price catalog: costs come from this file's prices.
    "MLFLOW_MODEL_CATALOG_URI": "",
    # No hint addressed to coding agents in the commands' output.
    "MLFLOW_DISABLE_AGENT_HINT": "1",
    # No job runner in `mlflow server`: on a sqlite store it would poll the store every minute
    # and could run LLM-judge jobs (online scoring). Read only by a server: when it starts, and
    # when a job is submitted, which it then refuses.
    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
}
# Set by mlflow_env() only when the environment leaves them unset: unless told otherwise,
# MLflow records no usage event for its developers.
MLFLOW_ENV_DEFAULTS: dict[str, str] = {"MLFLOW_DISABLE_TELEMETRY": "true"}
# The variables of MLFLOW_ENV and MLFLOW_ENV_DEFAULTS that MLflow 3.16 reads once, when it is
# imported: `import mlflow` starts its telemetry client and prints its hint to coding agents
# unless they say otherwise. Importing tracking/ or tuning/, whose modules import mlflow, sets
# them first; mlflow_env() sets the others before MLflow uses them.
MLFLOW_IMPORT_ENV: tuple[str, ...] = ("MLFLOW_DISABLE_AGENT_HINT", "MLFLOW_DISABLE_TELEMETRY")
# The MLflow datasets of a method's dataset-level metrics, by part (the dev rows are out of
# fold): log_method logs each dataset's metrics once, on the method's logged model, with the
# part's sample ids from the split the freeze read.
MLFLOW_METRIC_DATASETS: dict[str, str] = {"dev": "dev-oof", "test": "test"}
# The evaluation datasets, one per part, in the tag-methods experiment: `tracking datasets` copies
# each part's problems into its dataset once, and `tracking evaluate` scores a frozen method's
# predictions on it, one trace per problem.
MLFLOW_DATASETS: dict[str, str] = {"dev": "codeforces-dev", "test": "codeforces-test"}
# The fields of a problem that each record of an evaluation dataset shows, after its sample id:
# every raw column a method reads, as load_dataset() cleans them. They are only shown: an
# evaluation run looks the frozen predictions up by sample id.
EVALUATION_INPUT_FIELDS: tuple[str, ...] = (
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
    "prob_desc_notes",
    "prob_desc_sample_inputs",
    "prob_desc_sample_outputs",
    "source_code",
)
# check_traces: the summed cost of an API method's traces must equal its part's summed
# total_cost_usd in per_problem.csv, within this relative tolerance (math.isclose).
EVALUATION_COST_REL_TOL = 1e-9
# The MLflow UI screenshots of notebook 05 (notebooks/figures/mlflow_ui.py), dev only (ADR 0001).
# They choose no prediction, so none of these settings is in RECORDED_SETTINGS: one there would
# change every method's recipe. The capture's own `mlflow server` listens on 127.0.0.1 at this
# port: not MLflow's default 5000, where a user's `mlflow ui` would answer (the capture also
# refuses a port in use).
MLFLOW_UI_PORT = 5517
# The dev problem of the record and trace drawers: notebook 05's request example (section 1) and
# notebook 04's traced problem, whose `math` descode2tag-json misses.
MLFLOW_SCREEN_SAMPLE_ID = 0
# The zero-shot method shown, and compared with tfidf: the best dev focus macro F1 of the four
# (notebook 05, section 4).
MLFLOW_SCREEN_COMPARE = "descode2tag-json"
# The focus tag of the correct_<tag> = False filter: the zero-shot methods name `math` for most
# number theory problems (notebook 05, section 4).
MLFLOW_SCREEN_TAG = "math"
MLFLOW_SCREENS_DIR = PROJECT_ROOT / "notebooks" / "figures" / "mlflow"
# The MLflow UI screenshots of the search over Jev's words (mlflow_ui.py --set gepa, notebook 10,
# docs/adr/0010), in a folder of their own. The focus tag whose criteria the third view compares,
# version 1 (the seed's) against the chosen words': one whose criteria the chosen words changed
# (tuning/params.json), set before the capture. Like the settings above, it is in no recipe.
MLFLOW_SCREEN_GEPA_TAG = "math"
MLFLOW_SCREEN_GEPA_FOLDER = MLFLOW_SCREENS_DIR / "gepa"

# LLM toolkit (codeforces_tags.llm) -----------------------------------------------------------

# Transport settings, shared by every prompted method: how requests are sent, never what shapes
# an answer. They stay out of a request's hash (llm.settings.LlmSettings holds what is in it),
# so changing one never invalidates a recorded answer. A method's own settings live in its
# package. No response is stored at the provider: OpenAI keeps a stored response 30 days or more.
LLM_STORE = False
# "explicit" sends no cache breakpoint. Otherwise OpenAI writes a prompt of 1,024 tokens or more
# to its cache at 1.25 times the input price, and these prompts are never reused.
LLM_PROMPT_CACHE_MODE = "explicit"
LLM_SERVICE_TIER = "default"
# Requests in flight at once, then what the provider's SDK does when one fails: it retries (rate
# limits and timeouts too) with backoff, at most LLM_MAX_RETRIES times. The timeout bounds each
# attempt, so one request can take about (LLM_MAX_RETRIES + 1) * LLM_TIMEOUT_S seconds plus the
# backoff. All within the lowest usage tier. A request that still fails writes no line: the next
# `llm run` sends it again.
LLM_CONCURRENCY = 16
LLM_MAX_RETRIES = 5
LLM_TIMEOUT_S = 60.0
# `llm run` prints the answers so far, their tokens and their cost after every this many answers.
LLM_PROGRESS_EVERY = 100
# USD per million tokens, by model (a model of any provider adds a line): the price of uncached
# input tokens, of cached input tokens and of output tokens. gpt-6-luna is from OpenAI's pricing,
# which agrees with MLflow's upstream price table of 2026-09-23 (the 3.16.1 release's table lacks
# it, so every cost comes from here). jev-1.13.0 is from TypeSafe's models page of 2026-10-08: it
# bills input tokens only, its output is free, and its SDK reports no cached tokens, so the cached
# price is the input price. A frozen per_problem.csv holds costs computed from these prices, and
# every `verify` computes them again: change a price after the paid runs and it fails.
LLM_PRICES: dict[str, dict[str, float]] = {
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "jev-1.13.0": {"input": 0.042, "cached_input": 0.042, "output": 0.0},
}
# The version of the fixed parser of free-text answers (llm.answers.parse_text_tags), held by
# every run record. It was written and tested before any paid request and is never adjusted
# after results are seen: a new rule would be version 2.
TEXT_PARSER_VERSION = 1

# Serving (codeforces_tags.serving) -----------------------------------------------------------

# A frozen tagging method served over HTTP by MLflow's model server, in one Docker image per
# method. None of these settings shapes a prediction, so none is in RECORDED_SETTINGS. The build
# context of a method's image, build/serving/<method>/ (its MLflow model, the weights its method
# folder pins, its locked requirements and the Dockerfile), and the Kubernetes overlays that
# `serving manifests` writes, build/k8s/<method>-<device>/, live under the gitignored build/.
SERVING_BUILD_DIR = PROJECT_ROOT / "build" / "serving"
SERVING_K8S_DIR = PROJECT_ROOT / "build" / "k8s"
# A method's image is codeforces-tags/<method>, tagged with the git short sha (<sha>-dirty when
# built from uncommitted code) and latest.
SERVING_IMAGE_PREFIX = "codeforces-tags"
# The port of the model server in its container: nginx, in front of MLflow's scoring server.
SERVING_PORT = 8080
# Where the Streamlit app and `serving smoke` find the model server: --url, else
# $CODEFORCES_SERVING_URL (read at call time; an empty value counts as unset), else this URL.
SERVING_DEFAULT_URL = "http://127.0.0.1:8080"
SERVING_URL_ENV_VAR = "CODEFORCES_SERVING_URL"
# `serving smoke` predicts the first this many problems of the test set, one request each, and
# compares them with the method's frozen test rows.
SERVING_SMOKE_COUNT = 20
# The seconds the client waits for one answer: nginx and SageMaker cut a request at 60.
SERVING_TIMEOUT_S = 60
# The seconds a local model server may take to load its method and answer /ping.
SERVING_START_TIMEOUT_S = 180
# The seconds a stopped server's process group has between SIGTERM and SIGKILL: a local model
# server (serving.local.stop), and the MLflow UI and the Streamlit app that `serving up` started.
SERVING_STOP_TIMEOUT_S = 10
# `serving up` starts a method's model server, the MLflow UI and the Streamlit app in one
# terminal, and stops all three on Ctrl-C. The app (this script, run from the repository root,
# where Streamlit reads .streamlit/config.toml) listens on this port of 127.0.0.1, Streamlit's
# own default; the logs of the UI and of the app go to build/serving/<method>/<this folder>/.
SERVING_APP_PORT = 8501
SERVING_APP_SCRIPT = PROJECT_ROOT / "apps" / "streamlit_app.py"
SERVING_UP_LOG_DIR = "up"
# The compose project of the model server that `serving up` starts, followed by its method and
# port (codeforces-tags-up-tfidf-8080): each `up` drives a project of its own, so that it never
# recreates or removes a container that another `up`, or a compose run by hand, started.
SERVING_UP_COMPOSE_PROJECT = "codeforces-tags-up"
# How often `serving up` checks that each part still runs (a process that exited, a container
# that no longer runs), and the seconds the MLflow UI it starts may take to answer /health.
SERVING_UP_CHECK_INTERVAL_S = 3
SERVING_UP_UI_TIMEOUT_S = 60
# The MLflow UI that the Streamlit app links each served method to (the tag-methods experiment,
# the method's run, its prompt versions), where `serving mlflow-ui` serves the project's store;
# $CODEFORCES_MLFLOW_UI_URL, read at call time, overrides it when it is not empty. Not
# MLFLOW_UI_PORT, which the screenshot tool owns and refuses when in use.
SERVING_MLFLOW_UI_URL = "http://127.0.0.1:5000"
SERVING_MLFLOW_UI_URL_ENV_VAR = "CODEFORCES_MLFLOW_UI_URL"
# The variables a local `mlflow server` never inherits, from the screenshot tool
# (notebooks/figures/mlflow_ui.py) and from `serving mlflow-ui`: a regular expression searched in
# each variable's name, in any case. Credentials, with which the server's LLM features (its
# gateway, its judges) could spend money with a provider or a cloud (API keys, tokens, secrets,
# passwords, and credentials files such as GOOGLE_APPLICATION_CREDENTIALS), and proxies, through
# which its requests to 127.0.0.1 would leave the machine.
MLFLOW_SERVER_LEFT_OUT = (
    r"(?i)(_API_KEY|_TOKEN|_SECRET|_SECRET_ACCESS_KEY|_ACCESS_KEY_ID|_PASSWORD|_CREDENTIALS"
    r"|_PROXY)$"
)
