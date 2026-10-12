"""Settings of the TypeSafe Jev methods: what their requests and recipes hash (docs/adr/0007).

Every word Jev reads (the convention, the question, each tag's definition) and the model are in
every request's hash (``questions.JevRequest.sha256``) and in each method's recipe, with its
name. The two frozen methods read the words of the versioned prompt file
``templates/jev-tags.v1.yaml`` (docs/adr/0009), which ``SETTINGS`` holds; the searched method
reads the words its search chose from its frozen files (``words.words_of``, docs/adr/0010). A
calibrated twin reads its base's words and answers, and hashes its base's settings with its
threshold rule (docs/adr/0011). The transport settings are in neither hash, so changing one
invalidates no answer. A changed word after the first dev request is a new method under a new
name (docs/adr/0003): a new version of the prompt file, or a new search. The search's own
settings are recorded in its frozen ``tuning/params.json``.
"""

from codeforces_tags.llm.settings import LlmSettings
from codeforces_tags.prompt_files import PromptFile, prompt_file_path, read_prompt_file
from codeforces_tags.templating import render_string

# The registry keys of the frozen methods, whose words are the prompt file's below, and of the
# method whose words a search chose (docs/adr/0010), starting from those of GEPA_SEED_NAME.
FROZEN_NAMES: tuple[str, ...] = ("jev-statement", "jev-statement-code")
GEPA_NAME = "jev-statement-code-gepa"
GEPA_SEED_NAME = "jev-statement-code"
# The calibrated twins (docs/adr/0011), each by its base: a twin reads its base's committed
# answers and chooses its thresholds by the Platt + Lipton threshold rule. It sends no request of
# its own, and a prediction asks Jev its base's request.
PLATT_BASES: dict[str, str] = {
    "jev-statement-code-platt": "jev-statement-code",
    "jev-statement-code-gepa-platt": GEPA_NAME,
}
PLATT_NAMES: tuple[str, ...] = tuple(PLATT_BASES)
NAMES: tuple[str, ...] = (*FROZEN_NAMES, GEPA_NAME, *PLATT_NAMES)
# The provider and the model that answer them. The model is a pinned version, never the alias
# jev-latest (the SDK's default), which can move.
PROVIDER = "typesafe"
MODEL = "jev-1.13.0"
# The fields of each method's state, in order: the statement column of raw_to_problems, then, for
# jev-statement-code and the method searched from it, the solution (source_code). A calibrated
# twin's state is its base's.
STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "jev-statement": ("statement",),
    "jev-statement-code": ("statement", "solution"),
    GEPA_NAME: ("statement", "solution"),
}
STATE_FIELDS |= {twin: STATE_FIELDS[base] for twin, base in PLATT_BASES.items()}
# The methods compared in one study: none asks its test part until the dev answers of every
# method of its study are committed and complete (docs/adr/0003). The frozen methods are one
# study; the searched method is its own, compared with its seed on the held-out fold. The
# calibrated twins join no study: they send no request.
STUDIES: tuple[tuple[str, ...], ...] = (FROZEN_NAMES, (GEPA_NAME,))
# What a calibrated twin's run record discloses (docs/adr/0011): its threshold rule was proposed
# after a test score was read, so the command line's rule never names it eligible (ADR 0001).
TEST_INFORMED: dict[str, object] = {
    "notebook": "notebooks/10_jev_gepa.ipynb",
    "commits": ["8b9c456", "5a8366b"],
    "note": "the rule was proposed after the test drop of jev-statement-code-gepa was read",
}
# The words the frozen methods read (the convention, the question and each focus tag's
# definition) live in the versioned prompt file templates/jev-tags.v1.yaml, frozen since the first
# dev request: a new wording is a new version, read by new methods under new names (docs/adr/0003,
# docs/adr/0007). The definitions were written from Codeforces conventions, without reading a dev
# problem or answer. They are the seed of the search over Jev's words (words.SEED_WORDS).
PROMPT_NAME = "jev-tags"
PROMPT_VERSION = 1
PROMPT_PATH = prompt_file_path(PROMPT_NAME, PROMPT_VERSION)
PROMPT: PromptFile = read_prompt_file(
    __package__, PROMPT_PATH, {"convention", "question", "tag_definitions"}
)
# What every question tells Jev about Codeforces tags.
CONVENTION: str = PROMPT.data["convention"]
# The question asked for each focus tag: a Jinja2 template, rendered with the tag.
QUESTION_TEMPLATE: str = PROMPT.data["question"]
# The same question in the str.format form ("{tag}") that the frozen recipes hashed, before the
# words moved to the prompt file: SETTINGS keeps it, so both recipe_sha256 values hold, and so do
# the searched method's words, whose question is this one (words.JevWords).
QUESTION = render_string(QUESTION_TEMPLATE, tag="{tag}")
# Each focus tag's definition, in FOCUS_TAGS order (docs/adr/0007): what counts as a yes (a
# definition and what it includes) and as a no (a definition and what it excludes). A plain dict,
# as the file holds it.
TAG_DEFINITIONS: dict[str, dict[str, dict[str, object]]] = PROMPT.data["tag_definitions"]
# The search over Jev's words (docs/adr/0010). It never reads the held-out fold, learns from the
# learning folds and chooses on the selection fold.
SEARCH_HELD_OUT_FOLD = 0
SEARCH_SELECTION_FOLD = 4
SEARCH_LEARNING_FOLDS: tuple[int, ...] = (1, 2, 3)
# Each iteration's minibatch: this many learning-fold problems, first this many positives of each
# focus tag.
SEARCH_MINIBATCH_SIZE = 40
SEARCH_POSITIVES_PER_TAG = 2
# A reflection shows the reflection model this many problems, their texts cut to these lengths.
SEARCH_EXAMPLES = 8
SEARCH_STATEMENT_CHARS = 3_000
SEARCH_SOLUTION_CHARS = 2_000
# A rewritten piece holds at most this many times the characters of the seed's piece
# (words.length_cap), and words that share a sequence of this many words with a problem's text,
# absent from the seed words, are refused (words.copied_sequences).
SEARCH_LENGTH_FACTOR = 1.5
SEARCH_NGRAM = 8
# The search stops after this many iterations, or once Jev and the reflection model cost the
# cap of tune together (this many USD by default), and draws its minibatches from this seed.
SEARCH_MAX_ITERATIONS = 200
SEARCH_MAX_COST_USD = 15.0
SEARCH_SEED = 0
# The reflection model: GPT-6 Luna at high effort, through the LLM toolkit's OpenAI adapter, its
# answer under a strict JSON schema.
REFLECTION_SETTINGS = LlmSettings(
    provider="openai",
    model="gpt-6-luna",
    reasoning_effort="high",
    temperature=None,
    top_p=None,
    max_output_tokens=32_000,
    output_format="json",
)
# The options of gepa.optimize, written out: no merge, Pareto selection of the parent, the pieces
# in turn, a child accepted only when it beats its parent on the minibatch and then scored on the
# whole selection fold, and an error raised.
SEARCH_GEPA_OPTIONS: dict[str, object] = {
    "candidate_selection_strategy": "pareto",
    "module_selector": "round_robin",
    "acceptance_criterion": "strict_improvement",
    "val_evaluation_policy": "full_eval",
    "frontier_type": "instance",
    "use_merge": False,
    "skip_perfect_score": True,
    "perfect_score": 1.0,
    "seed": SEARCH_SEED,
    "raise_on_exception": True,
}
# Every setting of the search, as JSON values: recorded in tuning/params.json. The searched
# method's recipe hashes the words the search chose, not these.
SEARCH_SETTINGS: dict[str, object] = {
    "held_out_fold": SEARCH_HELD_OUT_FOLD,
    "selection_fold": SEARCH_SELECTION_FOLD,
    "learning_folds": list(SEARCH_LEARNING_FOLDS),
    "minibatch_size": SEARCH_MINIBATCH_SIZE,
    "positives_per_tag": SEARCH_POSITIVES_PER_TAG,
    "examples": SEARCH_EXAMPLES,
    "statement_chars": SEARCH_STATEMENT_CHARS,
    "solution_chars": SEARCH_SOLUTION_CHARS,
    "length_factor": SEARCH_LENGTH_FACTOR,
    "ngram": SEARCH_NGRAM,
    "max_iterations": SEARCH_MAX_ITERATIONS,
    "seed": SEARCH_SEED,
    "reflection": REFLECTION_SETTINGS.to_dict(),
    "gepa": dict(SEARCH_GEPA_OPTIONS),
}
# The search in MLflow: its experiment, its parent run, and the prefix of each piece's prompt.
SEARCH_EXPERIMENT = "jev-gepa"
SEARCH_RUN_NAME = f"search-{GEPA_NAME}"
SEARCH_PROMPT_PREFIX = "jev-gepa-"
# The search's cache, under config.EXPERIMENTS_DIR / GEPA_NAME / SEARCH_FOLDER (gitignored): Jev's
# answers, one file per set of words, the reflection model's answers, the search's own record and
# the id of its MLflow run.
SEARCH_FOLDER = "search"
JEV_CACHE_FOLDER = "jev"
REFLECTIONS_FILE = "reflections.jsonl"
HISTORY_LINES_FILE = "history.jsonl"
CANDIDATE_LINES_FILE = "candidates.jsonl"
OUTCOME_FILE = "outcome.json"
RUN_ID_FILE = "run.json"
# The frozen record of the search, in assets/methods/GEPA_NAME/TUNING_FOLDER, written by export.
TUNING_FOLDER = "tuning"
PARAMS_FILE = "params.json"
CANDIDATES_FILE = "candidates.csv"
HISTORY_FILE = "history.csv"
CANDIDATE_WORDS_FOLDER = "candidates"
# The dev-only record of a calibrated twin's threshold rule, in assets/methods/<twin>/TUNING_FOLDER,
# written by export: the rule and its inputs, the rules compared, the nested choice among them
# and the stability of their fold thresholds.
PLATT_FILES: tuple[str, ...] = (PARAMS_FILE, "rules.csv", "nested.csv", "stability.csv")
# The searched method's words, in its method folder.
WORDS_FILE_NAME = "words.json"
# The method folder's own file: the method's name, model, settings hash and thresholds.
METHOD_FILE_NAME = "jev.json"

# Transport: in no request hash, in no recipe. The SDK retries a timeout, a lost connection, a 408,
# a 429 and a 5xx at most MAX_RETRIES times within its own 30 s budget per call (RetryPolicy);
# TIMEOUT_S bounds each attempt. TypeSafe's limits are 80 requests and 100K tokens per second.
TIMEOUT_S = 60.0
MAX_RETRIES = 5
CONCURRENCY = 16
# The default cost cap of `run`, in USD: one method's whole dataset costs about 0.50.
MAX_COST_USD = 2.0
# `run` sends no new request after this many failed answers in a row. A request the API refuses
# costs nothing, so the cost cap never stops a run of them, and every failed line is final.
MAX_FAILED_IN_A_ROW = 20

# What each frozen method's recipe hashes, with its name (docs/adr/0004): the two share it, and
# it holds their state fields only, so it keeps the bytes it was frozen with. The searched
# method's recipe hashes its own words (words.method_settings). Its keys hold no dot, so MLflow's
# flattened parameters keep them apart.
SETTINGS: dict[str, object] = {
    "model": MODEL,
    "state_fields": {name: list(STATE_FIELDS[name]) for name in FROZEN_NAMES},
    "convention": CONVENTION,
    "question": QUESTION,
    "tag_definitions": TAG_DEFINITIONS,
}
