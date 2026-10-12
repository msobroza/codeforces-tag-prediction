"""Preprocessing step 3: read the Python solutions (Python 2 and 3)."""

from __future__ import annotations

import re

_IMPORT = re.compile(r"import\s+(.+)")  # "import sys, math as m"
_FROM_IMPORT = re.compile(r"from\s+(\S+)\s+import\b")  # "from math import*"
_PYTHON_VERSION = re.compile(r"(?:Python|PyPy) (\d)")


def imported_modules(source_code: str) -> set[str]:
    """List the top-level modules a solution imports.

    The code is scanned statement by statement instead of being parsed with ``ast``: about 19%
    of the solutions are Python 2 code that Python 3 cannot parse.

    Args:
        source_code: Python 2 or 3 source.

    Returns:
        Top-level module names, e.g. ``{"sys", "os"}`` for ``import sys, os.path``.
    """
    statements = (line.split("#")[0].strip() for line in re.split(r"[\n;]", source_code))
    names = [name for statement in statements for name in _imported_names(statement)]
    return {name.split(".")[0] for name in names if not name.startswith(".")}


def python_major_version(lang: str) -> int:
    """Read the Python major version from a runtime name.

    Args:
        lang: Runtime name such as ``"PyPy 3-64"`` or ``"Python 2"``.

    Returns:
        2 or 3.

    Raises:
        ValueError: If ``lang`` is not a Python runtime.
    """
    match = _PYTHON_VERSION.match(lang)
    if match is None:
        raise ValueError(f"Not a Python runtime: {lang!r}")
    return int(match.group(1))


def _imported_names(statement: str) -> list[str]:
    """Return the modules one statement imports, e.g. ``["sys", "os.path"]``, or ``[]``."""
    if match := _FROM_IMPORT.match(statement):
        return [match.group(1)]
    if match := _IMPORT.match(statement):
        # "sys, math as m" -> ["sys", "math"]
        return [name.split()[0] for name in match.group(1).split(",") if name.strip()]
    return []
