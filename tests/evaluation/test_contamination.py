"""Tests for the self-identifying solutions: solutions that name or quote their own problem."""

import subprocess

import pandas as pd
import pytest

from codeforces_tags.config import (
    PROJECT_ROOT,
    SELF_IDENTIFYING_FILE,
    SELF_IDENTIFYING_PATTERNS,
    SLICES_DIR,
)
from codeforces_tags.evaluation.contamination import (
    QUOTES_DESCRIPTION,
    RULE_SEPARATOR,
    SELF_IDENTIFYING_RULES,
    SLICE_COLUMNS,
    freeze_slice,
    load_slice,
    rule_matches,
    self_identifying_solutions,
)
from codeforces_tags.splitting import load_split

# Eight words: four distinct word 5-grams.
DESCRIPTION = "Alice sorts the array of n positive integers."


def make_problems(codes, description=DESCRIPTION):
    """Return one problem per solution, indexed by sample id from 0, all with one description."""
    return pd.DataFrame(
        {"prob_desc_description": description, "source_code": codes},
        index=pd.Index(range(len(codes)), name="sample_id"),
    )


def make_split(parts):
    """Return a split giving each problem, in order, its part; one group per problem."""
    index = pd.Index(range(len(parts)), name="sample_id")
    folds = [1 if part == "dev" else -1 for part in parts]
    return pd.DataFrame({"part": parts, "fold": folds, "group": range(len(parts))}, index=index)


def matched(code, description=DESCRIPTION):
    """Return the names of the rules one solution matches, in rule order."""
    row = rule_matches(make_problems([code], description)).iloc[0]
    return list(row.index[row])


def test_the_slice_is_frozen_under_assets_and_committed():
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(SELF_IDENTIFYING_FILE.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )

    assert SELF_IDENTIFYING_FILE.parent == SLICES_DIR
    assert SLICES_DIR.is_relative_to(PROJECT_ROOT / "assets")
    assert tracked.returncode == 0


def test_the_rules_are_the_patterns_then_the_quoted_description():
    assert SELF_IDENTIFYING_RULES == (
        "codeforces_url",
        "task_class",
        "round_or_id_header",
        "cf_problem_id",
        QUOTES_DESCRIPTION,
    )
    assert SELF_IDENTIFYING_RULES[:-1] == tuple(SELF_IDENTIFYING_PATTERNS)


def test_rule_matches_has_one_bool_column_per_rule_indexed_like_the_problems():
    problems = make_problems(["print(1)", "# codeforces.com/contest/1/problem/A"])

    matches = rule_matches(problems)

    assert list(matches.columns) == list(SELF_IDENTIFYING_RULES)
    assert matches.index.equals(problems.index)
    assert matches.dtypes.eq(bool).all()
    assert matches.sum(axis=1).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("code", "rules"),
    [
        ("# https://codeforces.com/contest/1228/problem/D", ["codeforces_url"]),
        ('"""http://codeforces.ru/problemset/problem/437/B"""', ["codeforces_url"]),
        ("# codeforces.com/gym/100001/problem/A", ["codeforces_url"]),
        ("# https://codeforces.com/profile/example_user", []),
        ("class CodeforcesTask355ASolution:\n    pass", ["task_class"]),
        ("class codeforcestask355a:\n    pass", []),
        ("#     Codeforces Round #488 by NEAR (Div. 2)", ["round_or_id_header"]),
        ("# Codeforces 1228D", ["round_or_id_header"]),
        ("# solved on codeforces", []),
        ("# cf 1228 D", ["cf_problem_id"]),
        ("#CF580B", ["cf_problem_id"]),
        ("# cf 1228 Z", []),
        ("x = abcf123A", []),
        (
            "# Codeforces Round #488\n# https://codeforces.com/contest/990/problem/A",
            ["codeforces_url", "round_or_id_header"],
        ),
    ],
)
def test_rule_matches_finds_a_solution_that_names_its_problem(code, rules):
    assert matched(code) == rules


@pytest.mark.parametrize(
    ("code", "rules"),
    [
        ("# Alice sorts the array of n\nprint(1)", [QUOTES_DESCRIPTION]),  # 2 of 4 5-grams
        ("# Alice sorts the array of\nprint(1)", []),  # 1 of 4
        ('"""ALICE SORTS THE ARRAY OF N POSITIVE INTEGERS"""', [QUOTES_DESCRIPTION]),
        ("'''alice sorts the array of n positive'''", [QUOTES_DESCRIPTION]),
        ('print("alice sorts the array of n positive integers")', []),  # code, not a comment
    ],
)
def test_rule_matches_finds_comments_that_quote_half_the_description(code, rules):
    assert matched(code) == rules


def test_rule_matches_never_flags_a_description_shorter_than_five_words():
    assert not matched("# sort the array\nprint(1)", description="Sort the array.")


def test_rule_matches_reads_a_missing_solution_or_description_as_empty_text():
    problems = make_problems([None, "# Alice sorts the array of n"], description=None)

    assert not rule_matches(problems).to_numpy().any()


def test_self_identifying_solutions_lists_the_flagged_problems_with_their_part_and_rules():
    codes = [
        "print(1)",
        "# Codeforces Round #488\n# https://codeforces.com/contest/990/problem/A",
        "print(3)",
        "class CodeforcesTask355ASolution:\n    pass",
    ]

    found = self_identifying_solutions(
        make_problems(codes), make_split(["test", "dev", "dev", "test"])
    )

    assert list(found.columns) == list(SLICE_COLUMNS) == ["part", "rules"]
    assert found.index.name == "sample_id"
    assert found.to_dict("index") == {
        1: {"part": "dev", "rules": "codeforces_url;round_or_id_header"},
        3: {"part": "test", "rules": "task_class"},
    }


def test_self_identifying_solutions_sorts_the_problems_by_sample_id():
    problems = make_problems(["# cf 11 A", "# cf 22 B"]).set_axis(
        pd.Index([9, 4], name="sample_id")
    )
    split = make_split(["dev", "test"]).set_axis(pd.Index([4, 9], name="sample_id"))

    found = self_identifying_solutions(problems, split)

    assert found.index.tolist() == [4, 9]
    assert found["part"].tolist() == ["dev", "test"]


def test_self_identifying_solutions_can_find_none():
    found = self_identifying_solutions(make_problems(["print(1)"]), make_split(["dev"]))

    assert found.empty
    assert list(found.columns) == ["part", "rules"]


def test_self_identifying_solutions_refuses_problems_other_than_the_split():
    with pytest.raises(ValueError, match="1 problems are not in the split"):
        self_identifying_solutions(make_problems(["a", "b"]), make_split(["dev"]))
    with pytest.raises(ValueError, match="1 split rows are not in the problems"):
        self_identifying_solutions(make_problems(["a"]), make_split(["dev", "test"]))


@pytest.fixture(name="frozen")
def frozen_fixture():
    """Return a slice of two problems."""
    problems = make_problems(["# cf 11 A", "print(2)", "# codeforces.com/contest/1/problem/A"])
    return self_identifying_solutions(problems, make_split(["dev", "dev", "test"]))


def test_freeze_slice_saves_the_slice_once(tmp_path, frozen):
    path = tmp_path / "slices" / "self_identifying_solutions.csv"

    message = freeze_slice(frozen, path)

    assert message == "saved 2 self-identifying problems to self_identifying_solutions.csv"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "sample_id,part,rules",
        "0,dev,cf_problem_id",
        "2,test,codeforces_url",
    ]
    assert load_slice(path).equals(frozen)


def test_freeze_slice_then_only_checks_the_file(tmp_path, frozen):
    path = tmp_path / "self_identifying_solutions.csv"
    freeze_slice(frozen, path)
    written = path.read_bytes()

    message = freeze_slice(frozen, path)

    assert message == "self_identifying_solutions.csv matches the slice (2 problems)"
    assert path.read_bytes() == written


def test_freeze_slice_refuses_to_replace_the_frozen_slice(tmp_path, frozen):
    path = tmp_path / "self_identifying_solutions.csv"
    freeze_slice(frozen, path)

    with pytest.raises(ValueError, match="no longer gives the frozen slice"):
        freeze_slice(frozen.iloc[:1], path)


def test_freeze_slice_round_trips_an_empty_slice(tmp_path):
    empty = self_identifying_solutions(make_problems(["print(1)"]), make_split(["dev"]))
    path = tmp_path / "self_identifying_solutions.csv"

    freeze_slice(empty, path)

    assert freeze_slice(empty, path).endswith("matches the slice (0 problems)")


def test_the_committed_slice_holds_the_66_self_identifying_problems():
    slice_ = load_slice()

    assert slice_["part"].value_counts().to_dict() == {"dev": 50, "test": 16}
    assert slice_["part"].equals(load_split()["part"].reindex(slice_.index))
    rules = slice_["rules"].str.split(RULE_SEPARATOR).explode().value_counts()
    assert rules.to_dict() == {
        "codeforces_url": 47,
        "task_class": 3,
        "round_or_id_header": 10,
        "cf_problem_id": 3,
        "quotes_description": 5,
    }
