"""Settings of the zero-shot methods, final from the first dev request (docs/adr/0003).

They reach every request's hash (the model, the sampling settings, the fields and constants a
prompt renders) and every method's recipe, so changing one makes a new method under a new
prompt name, never an edit.
"""

from codeforces_tags.llm.settings import OUTPUT_FORMATS
from codeforces_tags.prompt_files import prompt_file_path

# The provider and model the user chose. No dated snapshot of the model is known, so each
# answer records the model that served it.
PROVIDER = "openai"
MODEL = "gpt-6-luna"
# Always sent: the API's default effort is "medium", and "none" is the paper's direct prompting.
REASONING_EFFORT = "none"
# The authors' scripts (evaluation/program_synthesis/gen_program_synthesis.py and
# evaluation/apr/gen_apr.py) draw their temperatures from np.linspace(0, 2, 20); this is its
# point 3, float(np.linspace(0, 2, 20)[3]). The paper's "0.325" misreports it. The API takes a
# temperature only at reasoning effort "none", and `llm run` stops if it is rejected.
TEMPERATURE = 0.3157894736842105
# The authors' value, written as a float: a request's hash tells 1 from 1.0.
TOP_P = 1.0
# Our choice, the authors set no limit. It is far above the longest answer, which names all 37
# tags; an answer cut off at the cap fails (docs/adr/0003).
MAX_OUTPUT_TOKENS = 512

# The paper's two zero-shot settings, which are also the names of their prompts in the prompt
# registry: prompts:/code2tag/1 is its text version and prompts:/code2tag/2 its json version.
SETTINGS = ("code2tag", "descode2tag")
# One method per setting and output format, in the order the versions are registered. Fixed
# before the first test request: a method is neither added nor renamed afterwards.
METHOD_NAMES = tuple(
    f"{setting}-{output_format}" for setting in SETTINGS for output_format in OUTPUT_FORMATS
)
# The allow-list: the problem fields each setting's template reads, in the order its prompt
# prints them. Nothing else reaches a prompt, the labels least of all.
FIELDS = {
    "code2tag": ("source_code",),
    "descode2tag": (
        "prob_desc_description",
        "prob_desc_input_spec",
        "prob_desc_output_spec",
        "prob_desc_notes",
        "source_code",
    ),
}
# The dataset's lang_cluster column is not read: the templates' {{ lang_cluster }} renders this.
LANGUAGE = "Python"
# The name a json request sends its schema under, in text.format.
SCHEMA_NAME = "problem_tags"
# The version of a setting's prompt that each output format asks with, in OUTPUT_FORMATS order:
# the order in which register_prompts registers them, so a prompt file's version is its version
# in the prompt registry.
PROMPT_VERSIONS = {
    output_format: version for version, output_format in enumerate(OUTPUT_FORMATS, start=1)
}
# The prompts, one versioned prompt file per setting and output format, as POSIX paths relative
# to this package: the toolkit reads them from the package of the registered module. Version 1
# holds the template, and version 2 the same template and the answer schema.
PROMPT_PATHS = {
    setting: {
        output_format: prompt_file_path(setting, version)
        for output_format, version in PROMPT_VERSIONS.items()
    }
    for setting in SETTINGS
}

# Table 3 of the paper (docs/xcodeeval-paper-notes.md, which a test pins them to): all-tag macro
# F1 in percent, for Python, with gpt-3.5-turbo-0301. Ours is a fraction of 1, so notebooks print
# 100 times ours beside these.
PAPER_ALL_TAG_MACRO_F1 = {"code2tag": 29.45, "descode2tag": 33.73}
