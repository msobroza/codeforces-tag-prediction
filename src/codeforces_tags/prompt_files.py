"""Read a versioned prompt file: one YAML file per prompt and version, in a method's package.

A prompt file sits at ``templates/<name>.v<version>.yaml`` in its package, and holds its own
``name`` and ``version`` beside the keys its method reads (e.g. the template and its
settings). Its text and the sha256 of its bytes are kept, so that a frozen result can pin the
exact file it was computed from. The file is parsed with YAML's safe loader, which builds only
plain data, and a key written twice, at any depth, is refused instead of silently keeping the
last value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

_ALWAYS_READ = frozenset({"name", "version"})


@dataclass(frozen=True)
class PromptFile:
    """One version of a prompt, as its file holds it.

    Attributes:
        name: The prompt's name.
        version: The prompt's version, from 1.
        data: The file's other top-level keys, parsed.
        text: The file's text.
        sha256: The sha256 hex digest of the file's UTF-8 bytes.
    """

    name: str
    version: int
    data: dict[str, Any]
    text: str
    sha256: str


class _UniqueKeyLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    """YAML's safe loader, refusing a mapping that holds a key twice."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Build a mapping, refusing a key written twice.

        Args:
            node: The mapping's node.
            deep: Whether to build its values at once.

        Returns:
            The mapping.

        Raises:
            yaml.constructor.ConstructorError: If a key is written twice.
        """
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) < len(node.value):
            keys = [self.construct_object(key_node, deep=deep) for key_node, _ in node.value]
            duplicate = next(key for i, key in enumerate(keys) if key in keys[:i])
            raise yaml.constructor.ConstructorError(
                "while building a mapping", node.start_mark, f"duplicate key {duplicate!r}"
            )
        return mapping


def prompt_file_path(name: str, version: int) -> str:
    """Return where a prompt's version is, relative to its package.

    Args:
        name: The prompt's name.
        version: The prompt's version.

    Returns:
        ``templates/<name>.v<version>.yaml``, a POSIX path.
    """
    return f"templates/{name}.v{version}.yaml"


def read_prompt_file(package: str, path: str, keys: Collection[str]) -> PromptFile:
    """Read and check a prompt file of a package.

    Args:
        package: The package's import name (a package, not a plain module).
        path: The file's POSIX path, relative to the package.
        keys: The top-level keys the file holds beside ``name`` and ``version``.

    Returns:
        The prompt file.

    Raises:
        FileNotFoundError: If the file is not in the package.
        ValueError: If ``keys`` holds ``name`` or ``version``; if the file is not valid YAML,
            writes a key twice or is not a mapping; if its top-level keys are not exactly
            ``name``, ``version`` and ``keys``; if its name is not a non-empty str or its
            version an int of at least 1; or if ``path`` is not
            ``prompt_file_path(name, version)``.
    """
    if _ALWAYS_READ & set(keys):
        raise ValueError(f"keys {sorted(keys)}: name and version are always read")
    where = f"{path} of {package}"
    raw = files(package).joinpath(path).read_bytes()
    text = raw.decode("utf-8")
    content = _parse(text, where)
    _check_keys(content, _ALWAYS_READ | set(keys), where)
    name, version = content["name"], content["version"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: its name must be a non-empty str, not {name!r}")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{where}: its version must be an int of at least 1, not {version!r}")
    if path != prompt_file_path(name, version):
        raise ValueError(f"{where}: {name} v{version} must be at {prompt_file_path(name, version)}")
    return PromptFile(
        name=name,
        version=version,
        data={key: value for key, value in content.items() if key not in _ALWAYS_READ},
        text=text,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _parse(text: str, where: str) -> dict[Any, Any]:
    """Parse a prompt file's text into its top-level mapping.

    Raises:
        ValueError: If the text is not valid YAML, writes a key twice or is not a mapping.
    """
    try:
        content = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader: plain data only
    except yaml.YAMLError as error:
        raise ValueError(f"{where} is not valid YAML: {error}") from error
    if not isinstance(content, dict):
        raise ValueError(f"{where} holds a {type(content).__name__}, not a mapping")
    return content


def _check_keys(content: dict[Any, Any], expected: set[str], where: str) -> None:
    """Check that a prompt file's top-level keys are exactly the expected ones.

    Raises:
        ValueError: Naming the missing keys and the unexpected ones.
    """
    missing = sorted(expected - content.keys())
    unexpected = sorted(map(str, content.keys() - expected))
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if unexpected:
        problems.append(f"unexpected {unexpected}")
    if problems:
        raise ValueError(f"{where}: top-level keys " + "; ".join(problems))
