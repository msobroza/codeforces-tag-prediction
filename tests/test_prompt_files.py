"""Tests for reading a versioned prompt file from a package."""

import dataclasses
import hashlib
import sys

import pytest

from codeforces_tags.prompt_files import PromptFile, prompt_file_path, read_prompt_file

# A made-up package, written to tmp_path, that holds the prompt files of each test.
PACKAGE = "prompt_file_fakes"
KEYS = ("template", "settings")
TEXT = (
    "name: tagger\n"
    "version: 2\n"
    "settings:\n"
    "  model: made-up-1\n"
    "  temperature: 0.0\n"
    "template: |\n"
    "  Tag this: {{ statement }}\n"
)


@pytest.fixture(name="write_file")
def write_file_fixture(tmp_path, monkeypatch):
    """Make the package importable, and return a writer of a file under its ``templates/``.

    The package is forgotten after the test: the next test's package lives in another folder.
    """
    package = tmp_path / PACKAGE
    (package / "templates").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(text, file_name="tagger.v2.yaml"):
        path = f"templates/{file_name}"
        (package / path).write_bytes(text.encode("utf-8"))
        return path

    yield write
    for name in [name for name in sys.modules if name.partition(".")[0] == PACKAGE]:
        del sys.modules[name]


def test_prompt_file_path_names_the_file_by_prompt_and_version():
    assert prompt_file_path("tagger", 2) == "templates/tagger.v2.yaml"


def test_read_prompt_file_returns_the_name_version_data_text_and_hash(write_file):
    path = write_file(TEXT)

    prompt = read_prompt_file(PACKAGE, path, KEYS)

    assert prompt == PromptFile(
        name="tagger",
        version=2,
        data={
            "settings": {"model": "made-up-1", "temperature": 0.0},
            "template": "Tag this: {{ statement }}\n",
        },
        text=TEXT,
        sha256=hashlib.sha256(TEXT.encode("utf-8")).hexdigest(),
    )


def test_a_prompt_file_is_frozen(write_file):
    prompt = read_prompt_file(PACKAGE, write_file(TEXT), KEYS)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.version = 3  # type: ignore[misc]


def test_read_prompt_file_hashes_the_exact_bytes_of_the_file(write_file):
    text = TEXT.replace("\n", "\r\n") + "# é\r\n"

    prompt = read_prompt_file(PACKAGE, write_file(text), KEYS)

    assert prompt.text == text
    assert prompt.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_read_prompt_file_refuses_a_file_missing_from_the_package(write_file):
    write_file(TEXT)

    with pytest.raises(FileNotFoundError):
        read_prompt_file(PACKAGE, "templates/tagger.v3.yaml", KEYS)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (TEXT + "version: 3\n", "duplicate key 'version'"),
        (TEXT.replace("  temperature: 0.0\n", "  model: other\n"), "duplicate key 'model'"),
        (TEXT + "template: [\n", "is not valid YAML"),
        ("- name\n- version\n", "holds a list, not a mapping"),
        ("name: !!python/name:os.system\nversion: 2\n", "is not valid YAML"),
    ],
    ids=["top-level duplicate", "nested duplicate", "malformed", "not a mapping", "python tag"],
)
def test_read_prompt_file_refuses_a_file_that_is_not_a_mapping_of_unique_keys(
    write_file, text, match
):
    path = write_file(text)

    with pytest.raises(ValueError, match=match):
        read_prompt_file(PACKAGE, path, KEYS)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (TEXT.replace("version: 2\n", ""), r"missing \['version'\]"),
        (TEXT + "notes: hello\n", r"unexpected \['notes'\]"),
        (
            TEXT.replace("name: tagger\n", "") + "extra: 1\n",
            r"missing \['name'\]; unexpected \['extra'\]",
        ),
    ],
    ids=["missing", "unexpected", "both"],
)
def test_read_prompt_file_refuses_other_top_level_keys(write_file, text, match):
    path = write_file(text)

    with pytest.raises(ValueError, match=match):
        read_prompt_file(PACKAGE, path, KEYS)


@pytest.mark.parametrize("keys", [("template", "name"), ("version",)])
def test_read_prompt_file_refuses_name_or_version_among_the_other_keys(write_file, keys):
    path = write_file(TEXT)

    with pytest.raises(ValueError, match="name and version are always read"):
        read_prompt_file(PACKAGE, path, keys)


@pytest.mark.parametrize(
    "name", ["''", "3", "[tagger]", "null"], ids=["empty", "int", "list", "null"]
)
def test_read_prompt_file_refuses_a_name_that_is_not_a_non_empty_str(write_file, name):
    path = write_file(TEXT.replace("name: tagger", f"name: {name}"))

    with pytest.raises(ValueError, match="its name must be a non-empty str"):
        read_prompt_file(PACKAGE, path, KEYS)


@pytest.mark.parametrize(
    "version", ["0", "-1", "true", "'2'", "2.0"], ids=["zero", "negative", "bool", "str", "float"]
)
def test_read_prompt_file_refuses_a_version_that_is_not_a_positive_int(write_file, version):
    path = write_file(TEXT.replace("version: 2", f"version: {version}"))

    with pytest.raises(ValueError, match="its version must be an int of at least 1"):
        read_prompt_file(PACKAGE, path, KEYS)


@pytest.mark.parametrize("file_name", ["tagger.v3.yaml", "other.v2.yaml", "tagger.v2.yml"], ids=str)
def test_read_prompt_file_refuses_a_file_named_after_another_prompt_or_version(
    write_file, file_name
):
    path = write_file(TEXT, file_name)

    with pytest.raises(ValueError, match="must be at templates/tagger.v2.yaml"):
        read_prompt_file(PACKAGE, path, KEYS)
