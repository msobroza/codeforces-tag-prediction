"""Tests for the audit's steps, run end to end on a small dataset with a fake judge."""

import json
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.near_duplicates.audit.judges import JudgeSettings
from codeforces_tags.near_duplicates.audit.pipeline import (
    build_pairs,
    merge,
    pass_prompts,
    read_raw,
    run_pass,
)
from codeforces_tags.near_duplicates.audit.prompts import PASSES

pytestmark = pytest.mark.usefixtures("small_audit")


def run_all_passes(folder, query, settings=None):
    """Run the three judging passes in order."""
    for pass_name in PASSES:
        run_pass(pass_name, folder, settings=settings, query=query)


def test_build_pairs_writes_pairs_and_first_batches(problems, tmp_path):
    pairs = build_pairs(problems, tmp_path, recall_check=True)

    assert pd.read_csv(tmp_path / "pairs_to_judge.csv")["pair_id"].tolist() == list(
        range(len(pairs))
    )
    assert len(pd.read_csv(tmp_path / "candidates.csv")) == 7
    texts = tmp_path / "texts"
    batched = [
        item for k in range(2) for item in json.loads((texts / f"batch_{k}.json").read_text())
    ]
    assert [item["pair_id"] for item in batched] == pairs["pair_id"].tolist()


def test_build_pairs_judges_only_stage_1_candidates_by_default(problems, tmp_path):
    pairs = build_pairs(problems, tmp_path)

    assert pairs[["a", "b", "tier"]].to_numpy().tolist() == [[100, 101, "A: word>=0.4"]]
    assert not (tmp_path / "candidates.csv").exists()  # no recall check, no signal matrices
    assert [path.name for path in (tmp_path / "texts").glob("*.json")] == ["batch_0.json"]


def test_build_pairs_replaces_the_batches_of_an_earlier_build(problems, tmp_path):
    build_pairs(problems, tmp_path, recall_check=True)  # 2 batches

    build_pairs(problems, tmp_path)  # 1 batch

    assert sorted(path.name for path in (tmp_path / "texts").glob("*.json")) == ["batch_0.json"]


def test_the_passes_and_merge_work_on_stage_1_candidates_alone(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path)

    run_all_passes(tmp_path, fake_query)
    judgments = merge(tmp_path)

    assert judgments[["sample_id_a", "sample_id_b"]].to_numpy().tolist() == [[100, 101]]
    assert "char_cosine" not in judgments  # only the word similarity is computed
    assert pd.read_csv(tmp_path / "confirmed_pairs.csv").columns[0] == "sample_id_a"


def test_dry_run_writes_prompts_without_judging(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)

    prompts = run_pass("first_pass", tmp_path, dry_run=True, query=fake_query)

    assert len(prompts) == 2
    assert (tmp_path / "prompts" / "first_pass_1.txt").read_text() == prompts[1]
    assert not fake_query.calls
    assert not (tmp_path / "judgments_raw.json").exists()


def test_the_three_passes_then_merge_rebuild_the_verdicts(problems, tmp_path, fake_query):
    pairs = build_pairs(problems, tmp_path, recall_check=True)

    run_all_passes(tmp_path, fake_query, JudgeSettings(effort="low"))
    judgments = merge(tmp_path)

    raw = read_raw(tmp_path)
    is_easy_hard = (pairs["a"] == 100) & (pairs["b"] == 101)  # word 0.88, judged different
    easy_hard = int(pairs.loc[is_easy_hard, "pair_id"].iloc[0])
    assert list(raw) == [
        "first",
        "second",
        "sent_to_second",
        "second_round_2",
        "sent_to_second_round_2",
    ]
    # A duplicate call below 0.8, a low-confidence call, and a non-duplicate call above 0.8.
    assert raw["sent_to_second"] == sorted({0, 1, easy_hard})
    asked = sorted(raw["sent_to_second"] + raw["sent_to_second_round_2"])
    assert asked == pairs["pair_id"].tolist()
    assert judgments.loc[judgments["final"] == "duplicate", "pair_id"].tolist() == [0]
    assert (tmp_path / "confirmed_pairs.csv").exists()


def test_judges_run_in_the_texts_folder_and_the_run_is_recorded(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)

    run_pass("first_pass", tmp_path, settings=JudgeSettings(effort="low"), query=fake_query)

    assert {call["cwd"] for call in fake_query.calls} == {str(tmp_path / "texts")}
    assert {call["effort"] for call in fake_query.calls} == {"low"}
    record = json.loads((tmp_path / "run_first_pass.json").read_text())
    assert record["options"]["allowed_tools"] == ["Read(./**)"]
    assert "output_format" not in record["options"]
    assert record["total_cost_usd"] == pytest.approx(0.02)
    answer = json.loads((tmp_path / "answers" / "first_pass_0.json").read_text())
    assert set(answer) >= {"prompt_sha256", "labels", "cost_usd", "tool_calls"}


def test_a_relative_folder_gives_absolute_paths_in_the_prompts(
    problems, tmp_path, fake_query, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    build_pairs(problems, Path("audit"), recall_check=True)

    prompts = run_pass("first_pass", Path("audit"), dry_run=True, query=fake_query)

    assert f"Read the JSON file {tmp_path / 'audit' / 'texts' / 'batch_0.json'} (" in prompts[0]


def test_a_failed_judge_is_judged_again_alone(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)

    async def drops_a_pair(prompt, options):
        async for message in fake_query(prompt, options):
            if "batch_1" in prompt and hasattr(message, "structured_output"):
                message.structured_output["labels"].pop()
            yield message

    with pytest.raises(RuntimeError, match="1 of 2 judges of first_pass failed"):
        run_pass("first_pass", tmp_path, query=drops_a_pair)
    assert (tmp_path / "answers" / "first_pass_0.json").exists()
    assert not (tmp_path / "judgments_raw.json").exists()

    fake_query.calls.clear()
    run_pass("first_pass", tmp_path, query=fake_query)

    assert len(fake_query.calls) == 1  # only the failed batch
    assert "first" in read_raw(tmp_path)


def test_a_judge_that_errors_is_reported_with_its_error(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)

    async def fails_on_batch_1(prompt, options):
        if "batch_1" in prompt:
            raise ConnectionError("authentication failed")
        async for message in fake_query(prompt, options):
            yield message

    with pytest.raises(RuntimeError, match="ConnectionError: authentication failed"):
        run_pass("first_pass", tmp_path, query=fails_on_batch_1)
    assert (tmp_path / "answers" / "first_pass_0.json").exists()


def test_passes_must_run_in_order(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)

    with pytest.raises(ValueError, match="Run the judging pass first_pass first"):
        run_pass("second_pass_round_2", tmp_path, query=fake_query)
    with pytest.raises(ValueError, match="Run the judging pass first_pass first"):
        merge(tmp_path)


def test_running_an_earlier_pass_again_drops_the_later_ones(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)
    run_all_passes(tmp_path, fake_query)

    run_pass("first_pass", tmp_path, query=fake_query)

    assert list(read_raw(tmp_path)) == ["first"]
    with pytest.raises(ValueError, match="second_pass_round_1 first"):
        merge(tmp_path)


def test_merge_rejects_a_pair_without_a_second_label(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)
    run_all_passes(tmp_path, fake_query)
    raw = read_raw(tmp_path)
    raw["second_round_2"].pop()
    (tmp_path / "judgments_raw.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="1 pairs lack a judge's label"):
        merge(tmp_path)


def test_round_1_prompt_lists_its_pairs_and_names_the_batch_files(problems, tmp_path, fake_query):
    build_pairs(problems, tmp_path, recall_check=True)
    run_pass("first_pass", tmp_path, query=fake_query)

    [(prompt, pair_ids)] = pass_prompts("second_pass_round_1", tmp_path)

    texts = tmp_path / "texts"
    assert pair_ids[:2] == [0, 1]
    assert f"{texts / 'batch_0.json'} ... {texts / 'batch_1.json'}" in prompt
    assert f"Label ONLY these pair_ids: [{','.join(map(str, pair_ids))}]." in prompt
