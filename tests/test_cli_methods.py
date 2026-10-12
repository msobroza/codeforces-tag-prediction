"""Tests for the methods as the command line sees them: resolving one, loading it, the listing.

Every test runs offline: an API method is a spec in a test registry, its dependency group is
installed or not as a temporary ``pyproject.toml`` says, and its key is set to a sentinel or
removed with ``monkeypatch``; no request is sent and no key is read. The ``methods`` command,
which prints the listing, is tested here too.
"""

import inspect
import json

import pandas as pd
import pytest

from codeforces_tags import cli, cli_methods
from codeforces_tags.config import (
    API_KEY_ENV_VARS,
    CLI_DEFAULT_METHOD,
    CLI_OFFLINE_METHOD,
    DATA_DIR_ENV_VAR,
    FOCUS_TAGS,
    METHOD_FORMAT_VERSION,
    METHODS_ASSETS_DIR,
    PRETRAINED_DIR_ENV_VAR,
    TRAINED_DIR_ENV_VAR,
)
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    load_run_record,
    stack_parts,
)
from codeforces_tags.evaluation.timing import Latency, latency_summary, load_latency, save_latency
from codeforces_tags.methods import METHOD_FILE, METHODS, MODEL_FOLDER, MethodSpec
from codeforces_tags.methods.modernbert.settings import PRETRAINED_FILE, TRAINED_FILE
from codeforces_tags.methods.modernbert.weights import missing_caches
from codeforces_tags.methods.typesafe_jev.settings import GEPA_NAME, PLATT_BASES, TEST_INFORMED

# A module that no test may import: a spec that names it proves that nothing was imported.
NEVER_IMPORTED = "codeforces_tags.no_such_module"
# The key of the TypeSafe methods, set to this sentinel or removed; never a real key.
KEY = API_KEY_ENV_VARS["llm-typesafe"]
SENTINEL = "sentinel-key-never-sent"


def write_pyproject(tmp_path, **groups):
    """Write a pyproject.toml whose dependency groups are installed (empty) or not.

    Args:
        tmp_path: The test's folder.
        groups: Group name (``_`` for ``-``) -> whether it counts as installed: an installed
            group is empty, the other needs a distribution that no environment has.

    Returns:
        The path of the file.
    """
    lines = ["[dependency-groups]"]
    for group, installed in groups.items():
        requirement = "" if installed else '"no-such-distribution-for-the-cli-tests>=1"'
        lines.append(f"{group.replace('_', '-')} = [{requirement}]")
    path = tmp_path / "pyproject.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_method_file(folder, name):
    """Write the method.json of a frozen method folder, as save_method does."""
    folder.mkdir(parents=True)
    stored = {"method": name, "format_version": METHOD_FORMAT_VERSION, "tags": list(FOCUS_TAGS)}
    (folder / METHOD_FILE).write_text(json.dumps(stored), encoding="utf-8")


def api_spec(module=NEVER_IMPORTED):
    """Return the spec of a method that calls TypeSafe's API, served by ``module``."""
    return MethodSpec(
        module=module, kind="score", requirements=(), groups=("llm-typesafe",), calls_api=True
    )


def local_spec(groups=()):
    """Return the spec of a local score method in ``groups``, whose module is never imported."""
    return MethodSpec(module=NEVER_IMPORTED, kind="score", requirements=(), groups=groups)


def committed_out_of_fold(name):
    """Tell whether a method is committed in assets/methods/ and eligible by its run record.

    It is not chosen on dev (ADR 0004) and was not proposed after a test score was read
    (docs/adr/0011).
    """
    directory = METHODS_ASSETS_DIR / name
    if not (directory / MODEL_FOLDER / METHOD_FILE).is_file():
        return False
    entries = load_run_record(directory).method_entries
    return entries.get("dev_in_sample") is not True and not entries.get("test_informed")


def freeze_fake_method(directory, predictions, record, max_s, challenge):
    """Write the frozen files of a fake method: method folder, predictions, run record, latency.

    Its predictions of the dev and test problems of ``challenge`` (``cli_challenge``) are its
    frozen predictions of each part.
    """
    write_method_file(directory / MODEL_FOLDER, directory.name)
    parts = {part: predictions.loc[challenge[part]] for part in ("dev", "test")}
    stack_parts(**parts).to_csv(directory / PREDICTIONS_FILE)
    (directory / RUN_RECORD_FILE).write_text(json.dumps(record.to_dict()), encoding="utf-8")
    seconds = pd.Series(
        [max_s / 2, max_s], index=pd.Index([1, 2], name="sample_id"), name="seconds"
    )
    save_latency(directory, Latency(seconds, bulk_s=0.1, load_s=0.25))


@pytest.fixture(name="fake_methods")
def fake_methods_fixture(tmp_path, placeholder_record, cli_challenge):
    """Return a registry of six fake methods, their frozen folder and a pyproject.toml.

    ``exact`` and ``in_sample`` predict the truth of ``cli_challenge``, and so does ``slow``,
    whose slowest problem takes 12 s; ``in_sample`` records dev_in_sample. ``nothing`` predicts no
    tag, ``api`` calls TypeSafe's API and predicts no tag, and ``unfrozen`` has no file. The
    ``llm-typesafe`` group is installed.
    """
    assets, truth = tmp_path / "assets", cli_challenge["labels"]
    frozen = {
        "exact": (truth, False, 0.5),
        "in_sample": (truth, True, 0.5),
        "slow": (truth, False, 12.0),
        "nothing": (truth * 0, False, 0.5),
        "api": (truth * 0, False, 0.5),
    }
    for name, (predictions, in_sample, max_s) in frozen.items():
        entries = {"supersedes": [], "dev_in_sample": in_sample}
        record = placeholder_record(name, method_entries=entries)
        freeze_fake_method(assets / name, predictions, record, max_s, cli_challenge)
    registry = dict.fromkeys(["exact", "in_sample", "slow", "nothing", "unfrozen"], local_spec())
    registry["api"] = api_spec()
    pyproject = write_pyproject(tmp_path, llm_typesafe=True)
    return {"registry": registry, "assets_dir": assets, "pyproject": pyproject}


def test_offline_hint_names_the_offline_method_for_any_other_method():
    assert (
        cli_methods.offline_hint(CLI_DEFAULT_METHOD) == " (--method tfidf runs offline, for free)"
    )
    assert cli_methods.offline_hint(CLI_OFFLINE_METHOD) == ""


def test_resolve_method_returns_the_spec_of_a_frozen_method_that_runs_here():
    assert cli_methods.resolve_method(CLI_OFFLINE_METHOD) is METHODS[CLI_OFFLINE_METHOD]


def test_resolve_method_refuses_an_unregistered_name_and_lists_the_registry():
    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method("tf-idf")

    message = str(error.value)
    assert message.startswith("'tf-idf' is not a registered method: the registry has [")
    assert all(name in message for name in METHODS)
    assert message.endswith("(--method tfidf runs offline, for free).")


def test_resolve_method_refuses_a_method_that_is_not_frozen(tmp_path):
    registry = {"fresh": local_spec()}

    with pytest.raises(
        cli_methods.CliError, match=r"fresh is not frozen: .* holds no method\.json"
    ):
        cli_methods.resolve_method("fresh", registry=registry, assets_dir=tmp_path)


def test_a_method_missing_its_group_and_its_key_names_both_in_one_message(tmp_path, monkeypatch):
    # A user fixes both in one go, with the one command the message gives.
    monkeypatch.delenv(KEY, raising=False)
    pyproject = write_pyproject(tmp_path, llm_typesafe=False)

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method(CLI_DEFAULT_METHOD, pyproject=pyproject)

    assert str(error.value) == (
        "jev-statement-code cannot run here: it needs the dependency group llm-typesafe and "
        "TYPESAFE_API_KEY set. Run it as uv run --group llm-typesafe --env-file FILE "
        "codeforces-tags ..., FILE holding TYPESAFE_API_KEY=... (--method tfidf runs offline, "
        "for free)."
    )


@pytest.mark.parametrize("value", [None, ""])
def test_the_default_method_without_its_key_fails_before_loading_and_names_the_offline_method(
    tmp_path, monkeypatch, value
):
    # The SDK reads the key only when a session opens, after the folder loaded; the CLI checks
    # it first. The spec's module cannot be imported: resolving imports nothing.
    if value is None:
        monkeypatch.delenv(KEY, raising=False)
    else:
        monkeypatch.setenv(KEY, value)
    registry = {CLI_DEFAULT_METHOD: api_spec()}
    pyproject = write_pyproject(tmp_path, llm_typesafe=True)

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method(CLI_DEFAULT_METHOD, registry=registry, pyproject=pyproject)

    assert str(error.value) == (
        "jev-statement-code cannot run here: it needs TYPESAFE_API_KEY set. Run it as uv run "
        "--env-file FILE codeforces-tags ..., FILE holding TYPESAFE_API_KEY=... (--method tfidf "
        "runs offline, for free)."
    )


def test_a_method_whose_group_is_not_installed_fails_before_its_module_is_imported(tmp_path):
    # Otherwise load_method would import torch and fail with a traceback.
    registry = {"heavy": local_spec(groups=("finetuning",))}
    write_method_file(tmp_path / "assets" / "heavy" / MODEL_FOLDER, "heavy")
    pyproject = write_pyproject(tmp_path, finetuning=False)

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method(
            "heavy", registry=registry, assets_dir=tmp_path / "assets", pyproject=pyproject
        )

    assert str(error.value) == (
        "heavy cannot run here: it needs the dependency group finetuning. Run it as uv run "
        "--group finetuning codeforces-tags ... (--method tfidf runs offline, for free)."
    )


def test_resolve_method_names_only_the_groups_that_are_not_installed(tmp_path):
    registry = {"two": local_spec(groups=("finetuning", "llm-typesafe"))}
    write_method_file(tmp_path / "assets" / "two" / MODEL_FOLDER, "two")
    pyproject = write_pyproject(tmp_path, finetuning=True, llm_typesafe=False)

    with pytest.raises(
        cli_methods.CliError, match=r"needs the dependency group llm-typesafe\. Run it as"
    ):
        cli_methods.resolve_method(
            "two", registry=registry, assets_dir=tmp_path / "assets", pyproject=pyproject
        )


def test_resolve_method_accepts_an_api_method_whose_group_is_installed_and_key_set(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(KEY, SENTINEL)
    pyproject = write_pyproject(tmp_path, llm_typesafe=True)

    spec = cli_methods.resolve_method(CLI_DEFAULT_METHOD, pyproject=pyproject)

    assert spec is METHODS[CLI_DEFAULT_METHOD]


@pytest.fixture(name="weighted")
def weighted_fixture(tmp_path, monkeypatch):
    """Freeze a method whose folder holds a pretrained and a trained weight list, as ModernBERT's.

    It is registered with the spec of ``modernbert_lora``, its group is installed, and both
    caches are empty temporary folders.

    Returns:
        The keyword arguments of ``resolve_method`` and ``method_table``, and the two caches.
    """
    folder = tmp_path / "assets" / "bag" / MODEL_FOLDER
    write_method_file(folder, "bag")
    pretrained = {"repo_id": "org/encoder", "revision": "abc", "files": {"model.safetensors": "0"}}
    (folder / PRETRAINED_FILE).write_text(json.dumps(pretrained), encoding="utf-8")
    trained = {"root": "bag", "folds": {"fold_0": {"file": "fold_0.safetensors", "sha256": "0"}}}
    (folder / TRAINED_FILE).write_text(json.dumps(trained), encoding="utf-8")
    caches = {"pretrained": tmp_path / "pretrained", "trained": tmp_path / "trained"}
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(caches["pretrained"]))
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(caches["trained"]))
    return {
        "registry": {"bag": METHODS["modernbert_lora"]},
        "assets_dir": tmp_path / "assets",
        "pyproject": write_pyproject(tmp_path, finetuning=True),
    }, caches


def test_a_method_whose_weights_are_not_in_their_caches_does_not_run_here(weighted):
    # Regression: with its group installed, a ModernBERT method of a fresh clone showed runs_here
    # True, then failed to load: its weights are in neither cache.
    places, caches = weighted

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method("bag", **places)

    assert str(error.value) == (
        f"bag cannot run here: it needs the pretrained weights of org/encoder in "
        f"{caches['pretrained'] / 'org--encoder' / 'abc'} (fetch them: `uv run --group "
        "finetuning python -m codeforces_tags.methods.modernbert fetch --method bag`) and the "
        f"trained weights in {caches['trained'] / 'bag'} (copy bag/ from the trained cache of the "
        "machine that froze the method, docs/adr/0006) (--method tfidf runs offline, for free)."
    )
    table = cli_methods.method_table(None, **places)
    assert table.loc["bag", ["needs", "runs_here"]].tolist() == [
        "group finetuning, pretrained weights, trained weights",
        False,
    ]


def test_a_method_whose_weights_are_in_their_caches_runs_here(weighted):
    places, caches = weighted
    for path in (
        caches["pretrained"] / "org--encoder" / "abc" / "model.safetensors",
        caches["trained"] / "bag" / "fold_0.safetensors",
    ):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"w")

    assert cli_methods.resolve_method("bag", **places) is places["registry"]["bag"]


@pytest.fixture(name="f2llm_folder")
def f2llm_folder_fixture(tmp_path, monkeypatch):
    """Freeze a method whose pretrained.json lists ``{"path", "sha256"}`` entries, as F2LLM's.

    It is registered with the spec of ``f2llm_probe``, its group is installed, and the
    pretrained cache is an empty temporary folder.

    Returns:
        The keyword arguments of ``resolve_method`` and ``method_table``, and the snapshot folder
        that its weight list names.
    """
    folder = tmp_path / "assets" / "f2llm_probe" / MODEL_FOLDER
    write_method_file(folder, "f2llm_probe")
    files = [{"path": "model.safetensors", "sha256": "0"}, {"path": "1_Pooling/config.json"}]
    pretrained = {"repo_id": "org/encoder", "revision": "abc", "files": files}
    (folder / PRETRAINED_FILE).write_text(json.dumps(pretrained), encoding="utf-8")
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "pretrained"))
    return {
        "registry": {"f2llm_probe": METHODS["f2llm_probe"]},
        "assets_dir": tmp_path / "assets",
        "pyproject": write_pyproject(tmp_path, f2llm=True),
    }, tmp_path / "pretrained" / "org--encoder" / "abc"


def test_an_f2llm_method_whose_snapshot_is_in_the_cache_runs_here(f2llm_folder):
    # Regression: the command line read an F2LLM pretrained.json as ModernBERT's, and failed with a
    # TypeError on its {"path", "sha256"} entries.
    places, snapshot = f2llm_folder
    for name in ("model.safetensors", "1_Pooling/config.json"):
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"w")

    assert cli_methods.resolve_method("f2llm_probe", **places) is METHODS["f2llm_probe"]
    table = cli_methods.method_table(None, **places)
    assert table.loc["f2llm_probe", ["needs", "runs_here"]].tolist() == [
        "group f2llm, pretrained weights",
        True,
    ]


def test_an_f2llm_method_missing_a_snapshot_file_names_its_own_fetch_command(f2llm_folder):
    places, snapshot = f2llm_folder
    (snapshot / "model.safetensors").parent.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"w")

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.resolve_method("f2llm_probe", **places)

    assert str(error.value) == (
        f"f2llm_probe cannot run here: it needs the pretrained weights of org/encoder in "
        f"{snapshot} (fetch them: `uv run --group f2llm python -m codeforces_tags.methods.f2llm "
        "fetch --method f2llm_probe`) (--method tfidf runs offline, for free)."
    )


def test_a_corrupt_pretrained_json_stops_the_listing_rather_than_hiding_as_not_runnable(
    f2llm_folder,
):
    # Regression: a weight list in neither shape is a corrupt frozen file, so resolve_method and
    # method_table raise its ValueError, as their docstrings say; runs_here catches CliError only.
    places, _ = f2llm_folder
    folder = places["assets_dir"] / "f2llm_probe" / MODEL_FOLDER
    corrupt = {"repo_id": "org/encoder", "revision": "abc"}
    (folder / PRETRAINED_FILE).write_text(json.dumps(corrupt), encoding="utf-8")

    for check in (
        lambda: cli_methods.resolve_method("f2llm_probe", **places),
        lambda: cli_methods.method_table(None, **places),
    ):
        with pytest.raises(ValueError, match="its files are None"):
            check()


def test_package_command_runs_the_package_of_a_methods_module_with_each_of_its_groups():
    two_groups = MethodSpec(
        module="codeforces_tags.methods.pkg.method",
        kind="score",
        requirements=(),
        groups=("llm", "f2llm"),
    )

    assert cli_methods.package_command(METHODS["f2llm_probe"]) == (
        "uv run --group f2llm python -m codeforces_tags.methods.f2llm"
    )
    assert cli_methods.package_command(two_groups) == (
        "uv run --group llm --group f2llm python -m codeforces_tags.methods.pkg"
    )
    assert cli_methods.package_command(local_spec()) == "uv run python -m codeforces_tags"
    # A ModernBERT method's command is the one missing_caches names by default.
    default = inspect.signature(missing_caches).parameters["command"].default
    assert default == "uv run --group finetuning python -m codeforces_tags.methods.modernbert"
    for name in ("modernbert_lora", "modernbert_base_full"):
        assert cli_methods.package_command(METHODS[name]) == default


def test_load_cli_method_loads_the_frozen_method_folder():
    method = cli_methods.load_cli_method("label_prior")

    assert (method.name, method.tags) == ("label_prior", FOCUS_TAGS)


def test_load_cli_method_reports_an_import_error_with_the_offline_method(tmp_path):
    write_method_file(tmp_path / "ghost" / MODEL_FOLDER, "ghost")

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.load_cli_method("ghost", registry={"ghost": local_spec()}, assets_dir=tmp_path)

    message = str(error.value)
    assert message.startswith(f"ghost could not be loaded from {tmp_path / 'ghost' / 'model'}: ")
    assert "ModuleNotFoundError: No module named 'codeforces_tags.no_such_module'" in message
    assert message.endswith(" (--method tfidf runs offline, for free).")


def test_load_cli_method_reports_a_missing_file(tmp_path):
    with pytest.raises(cli_methods.CliError, match="FileNotFoundError: .* holds no method.json"):
        cli_methods.load_cli_method("ghost", registry={"ghost": local_spec()}, assets_dir=tmp_path)


def test_load_cli_method_reports_weights_that_differ_from_the_frozen_ones(tmp_path, monkeypatch):
    # A ModernBERT cache that holds other weights raises ChangedWeightsError, a ValueError.
    def changed(folder, **_):
        raise ValueError(f"['config.json'] in {folder} differ from the hashes the folder records.")

    monkeypatch.setattr(cli_methods, "load_method", changed)

    with pytest.raises(cli_methods.CliError) as error:
        cli_methods.load_cli_method("ghost", registry={"ghost": local_spec()}, assets_dir=tmp_path)

    folder = tmp_path / "ghost" / MODEL_FOLDER
    assert str(error.value) == (
        f"ghost could not be loaded from {folder}: ValueError: ['config.json'] in {folder} differ "
        "from the hashes the folder records (--method tfidf runs offline, for free)."
    )


def test_method_table_scores_times_and_ranks_each_method_and_says_what_it_needs(
    fake_methods, cli_challenge, monkeypatch
):
    monkeypatch.delenv(KEY, raising=False)

    table = cli_methods.method_table(cli_challenge["labels"], **fake_methods)

    assert table.index.name == "method"
    assert list(table.index) == ["exact", "in_sample", "slow", "nothing", "unfrozen", "api"]
    assert list(table.columns) == list(cli_methods.METHOD_TABLE_COLUMNS)
    assert table["dev_macro_f1"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert table["test_macro_f1"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert table.loc["exact", ["median_s", "max_s", "load_s", "cold_s"]].tolist() == [
        0.375,
        0.5,
        0.25,
        0.75,
    ]
    assert table.loc["unfrozen"].isna()[["dev_macro_f1", "max_s", "load_s", "cold_s"]].all()
    assert table["eligible"].tolist() == [True, False, False, True, False, True]
    assert table["best_dev"].tolist() == [True, False, False, False, False, False]
    assert not table["default"].any()
    assert table["needs"].tolist() == ["", "", "", "", "", "group llm-typesafe, TYPESAFE_API_KEY"]
    assert table["runs_here"].tolist() == [True, True, True, True, False, False]


def test_method_table_marks_no_method_best_without_labels(fake_methods, monkeypatch):
    monkeypatch.setenv(KEY, SENTINEL)

    table = cli_methods.method_table(None, **fake_methods)

    assert table[["dev_macro_f1", "test_macro_f1"]].isna().all().all()
    assert not table["best_dev"].any()
    assert table.loc["api", "runs_here"]


def test_method_table_lists_the_committed_methods_with_their_latency_and_the_default():
    table = cli_methods.method_table(None)

    assert list(table.index) == list(METHODS)
    assert table["kind"].tolist() == [spec.kind for spec in METHODS.values()]
    assert table.index[table["default"]].tolist() == [CLI_DEFAULT_METHOD]
    # ADR 0004: every committed method that was not chosen on dev data is frozen out of fold,
    # and answers in under 10 s. A method registered but never frozen (f2llm_setfit) has no
    # committed folder; the searched Jev method, whose words were chosen on dev folds 1 to 4
    # (docs/adr/0010), records dev_in_sample in its committed run record. Neither is eligible.
    assert table["eligible"].tolist() == [committed_out_of_fold(name) for name in METHODS]
    assert not table.loc[GEPA_NAME, "eligible"]
    summary = latency_summary(load_latency(METHODS_ASSETS_DIR / "tfidf").seconds)
    assert table.loc["tfidf", "max_s"] == summary["max_s"]
    assert table.loc[CLI_DEFAULT_METHOD, "needs"] == "group llm-typesafe, TYPESAFE_API_KEY"
    assert table.loc["modernbert_lora", "needs"] == (
        "group finetuning, pretrained weights, trained weights"
    )
    assert table.loc["tfidf", "needs"] == ""
    # ADR 0004 counts the slowest problem once loaded; one fresh predict also loads the method.
    assert table.loc[CLI_DEFAULT_METHOD, "cold_s"] < 1.0
    assert table.loc["modernbert_base_full", "cold_s"] > 10.0


def test_the_searched_jev_method_is_never_eligible_however_good_and_fast(
    tmp_path, cli_challenge, placeholder_record
):
    # Its freeze records dev_in_sample (docs/adr/0010): frozen, perfect on dev and test and
    # under the latency budget, it is still never eligible, so never best on dev.
    entries = {"supersedes": [], "dev_in_sample": True}
    record = placeholder_record(GEPA_NAME, method_entries=entries)
    freeze_fake_method(tmp_path / GEPA_NAME, cli_challenge["labels"], record, 0.5, cli_challenge)

    table = cli_methods.method_table(
        cli_challenge["labels"],
        registry={GEPA_NAME: METHODS[GEPA_NAME]},
        assets_dir=tmp_path,
        pyproject=write_pyproject(tmp_path, llm_typesafe=True),
    )

    row = table.loc[GEPA_NAME]
    assert row[["dev_macro_f1", "test_macro_f1", "max_s"]].tolist() == [1.0, 1.0, 0.5]
    assert (row["eligible"], row["best_dev"]) == (False, False)


def test_a_method_proposed_after_a_test_score_was_read_is_never_eligible(
    tmp_path, cli_challenge, placeholder_record
):
    # docs/adr/0011: out of fold, perfect on dev and test and fast, it is still never eligible.
    entries = {"dev_in_sample": False, "test_informed": TEST_INFORMED}
    record = placeholder_record("alone", method_entries=entries)
    freeze_fake_method(tmp_path / "alone", cli_challenge["labels"], record, 0.5, cli_challenge)

    row = alone_row(tmp_path, cli_challenge)

    assert row[["dev_macro_f1", "test_macro_f1", "max_s"]].tolist() == [1.0, 1.0, 0.5]
    assert (row["eligible"], row["best_dev"]) == (False, False)


@pytest.mark.parametrize("twin", list(PLATT_BASES))
def test_a_calibrated_twin_leaves_the_default_and_best_dev_to_the_other_methods(
    tmp_path, cli_challenge, placeholder_record, twin
):
    # The twin, frozen as its freeze records it, beats the default on dev and test; the default
    # stays the default, and the only eligible method is best on dev.
    truth = cli_challenge["labels"]
    frozen = {
        twin: (truth, {"dev_in_sample": False, "test_informed": TEST_INFORMED}),
        CLI_DEFAULT_METHOD: (truth.assign(math=0), {"dev_in_sample": False}),
    }
    for name, (predictions, entries) in frozen.items():
        record = placeholder_record(name, method_entries=entries)
        freeze_fake_method(tmp_path / name, predictions, record, 0.5, cli_challenge)

    table = cli_methods.method_table(
        truth,
        registry={name: METHODS[name] for name in frozen},
        assets_dir=tmp_path,
        pyproject=write_pyproject(tmp_path, llm_typesafe=True),
    )

    assert table.loc[twin, "dev_macro_f1"] > table.loc[CLI_DEFAULT_METHOD, "dev_macro_f1"]
    columns = ["eligible", "best_dev", "default"]
    assert table.loc[twin, columns].tolist() == [False, False, False]
    assert table.loc[CLI_DEFAULT_METHOD, columns].tolist() == [True, True, True]


def alone_row(tmp_path, cli_challenge):
    """Return the row of the method ``alone``, frozen in ``tmp_path``, listed by itself."""
    table = cli_methods.method_table(
        cli_challenge["labels"],
        registry={"alone": local_spec()},
        assets_dir=tmp_path,
        pyproject=write_pyproject(tmp_path),
    )
    return table.loc["alone"]


def test_method_table_scores_the_macro_f1_not_the_micro_f1(
    tmp_path, cli_challenge, placeholder_record
):
    # Math for every problem: math's F1 is 2/9 and every other tag's 0, so the macro F1 is 1/36,
    # while the micro F1 would be 1/8.
    record = placeholder_record("alone", method_entries={"dev_in_sample": False})
    math_only = (cli_challenge["labels"] * 0).assign(math=1)
    freeze_fake_method(tmp_path / "alone", math_only, record, 0.5, cli_challenge)

    row = alone_row(tmp_path, cli_challenge)

    assert row[["dev_macro_f1", "test_macro_f1"]].tolist() == [pytest.approx(1 / 36)] * 2


def test_a_method_whose_slowest_problem_takes_the_whole_budget_is_not_eligible(
    tmp_path, cli_challenge, placeholder_record
):
    record = placeholder_record("alone", method_entries={"dev_in_sample": False})
    freeze_fake_method(tmp_path / "alone", cli_challenge["labels"], record, 10.0, cli_challenge)

    row = alone_row(tmp_path, cli_challenge)

    assert (row["max_s"], row["eligible"]) == (10.0, False)


def test_a_frozen_method_without_its_run_record_is_not_eligible(
    tmp_path, cli_challenge, placeholder_record
):
    record = placeholder_record("alone", method_entries={"dev_in_sample": False})
    freeze_fake_method(tmp_path / "alone", cli_challenge["labels"], record, 0.5, cli_challenge)
    (tmp_path / "alone" / RUN_RECORD_FILE).unlink()

    row = alone_row(tmp_path, cli_challenge)

    assert (row["dev_macro_f1"], row["eligible"]) == (1.0, False)


def test_format_method_table_rounds_f1_to_four_decimals_and_seconds_to_three():
    table = pd.DataFrame(
        {"dev_macro_f1": [0.123456], "median_s": [0.37549], "load_s": [float("nan")]},
        index=pd.Index(["m"], name="method"),
    )

    assert cli_methods.format_method_table(table).splitlines()[-1].split() == [
        "m",
        "0.1235",
        "0.375",
        "-",
    ]


def test_methods_lists_every_method_and_notes_that_its_scores_need_the_dataset(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path / "missing"))

    status = cli.main(["methods"])

    output, errors = capsys.readouterr()
    assert status == 0
    assert all(name in output for name in METHODS)
    assert output.splitlines()[0].split()[:3] == ["kind", "dev_macro_f1", "test_macro_f1"]
    assert errors.startswith("codeforces-tags: note: no F1 scores without the challenge dataset:")
    assert "--data DIR" in errors
    assert len(errors.splitlines()) == 1


def test_methods_refuses_a_data_directory_given_that_is_not_the_dataset(tmp_path, capsys):
    # Regression: an explicit --data that is wrong is the user's error, not a missing dataset.
    status = cli.main(["methods", "--data", str(tmp_path / "missing")])

    output, errors = capsys.readouterr()
    assert (status, output) == (1, "")
    assert errors.splitlines() == [
        f"codeforces-tags: error: {tmp_path / 'missing'} does not exist: --data DIR must be a "
        "directory of sample_<id>.json files (default: $CODEFORCES_DATA_DIR)."
    ]


def test_methods_scores_the_frozen_predictions_on_the_dataset(
    cli_challenge, fake_methods, monkeypatch, capsys
):
    for name, value in [
        ("METHODS", fake_methods["registry"]),
        ("METHODS_ASSETS_DIR", fake_methods["assets_dir"]),
        ("PYPROJECT_FILE", fake_methods["pyproject"]),
    ]:
        monkeypatch.setattr(cli, name, value)

    status = cli.main(["methods", "--data", str(cli_challenge["data_dir"])])

    output, errors = capsys.readouterr()
    rows = {line.split()[0]: line.split()[1:] for line in output.splitlines()[2:]}
    assert (status, errors) == (0, "")
    assert rows["exact"][:7] == ["score", "1.0000", "1.0000", "0.375", "0.500", "0.250", "0.750"]
    assert rows["unfrozen"][1:7] == ["-"] * 6
