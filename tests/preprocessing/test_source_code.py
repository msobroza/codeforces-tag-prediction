"""Tests for the solution-code helpers."""

import pytest

from codeforces_tags.preprocessing.source_code import imported_modules, python_major_version


def test_imported_modules_reads_plain_aliased_dotted_and_from_imports():
    source = "\n".join(
        [
            "import sys, math as m",
            "from collections import deque",
            "import os.path",
            "    import heapq  # inside a function",
            "from . import sibling",
            "x = 'import fake'",
        ]
    )

    assert imported_modules(source) == {"sys", "math", "collections", "os", "heapq"}


def test_imported_modules_works_on_python2_code():
    source = "import sys; input = sys.stdin.readline\nprint 'hello'\nfrom fractions import gcd"

    assert imported_modules(source) == {"sys", "fractions"}


def test_imported_modules_handles_star_imports_without_a_space():
    assert imported_modules("from math import*\nfrom itertools import *") == {"math", "itertools"}


def test_imported_modules_ignores_names_that_only_start_with_import():
    assert imported_modules("importlib = 1\nfromage = 2") == set()


def test_imported_modules_finds_imports_after_a_semicolon():
    assert imported_modules("import sys; import math") == {"sys", "math"}
    assert imported_modules("import io,os;from math import *") == {"io", "os", "math"}


@pytest.mark.parametrize(
    ("lang", "expected"),
    [("Python 3", 3), ("PyPy 3-64", 3), ("PyPy 2", 2), ("Python 2", 2)],
)
def test_python_major_version(lang, expected):
    assert python_major_version(lang) == expected


def test_python_major_version_rejects_unknown_languages():
    with pytest.raises(ValueError, match="Kotlin"):
        python_major_version("Kotlin")
