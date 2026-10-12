"""The answers files: append-only JSON lines, both the resume cache and the record (docs/adr/0003).

``assets/methods/<name>/answers_<part>.jsonl`` holds one line per ``request_sha256``: an answer,
written when it arrives and never rewritten. A request that has a line is answered, failed or
not, and is never sent again; a request that has none is sent by the next ``llm run``. So a run
that is killed loses no answer, and the same lines are the record that every number of a
prompted method is a pure function of.

A line is an answer only when it ends with a newline. The last line of a killed run may have
no newline, a write cut short: ``read_answers`` reports and ignores it, and ``append_answer``
removes it before it writes, so a partial line never ends up inside the file. Any other line
that is not an answer is a corrupt file, and raises.

One ``llm run`` at a time: the functions do not lock the file.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from codeforces_tags.llm.answers import problem_list
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.methods import method_dir
from codeforces_tags.splitting import PARTS

_NEWLINE = b"\n"


def answers_file(name: str, part: str) -> Path:
    """Return the answers file of a method's part.

    Args:
        name: The method's registry key.
        part: ``dev`` or ``test``.

    Returns:
        ``method_dir(name) / answers_file_name(part)``; the file may not exist yet.

    Raises:
        ValueError: If the name is not a method name, or the part is not one of ``PARTS``.
    """
    file_name = answers_file_name(part)
    return method_dir(name) / file_name


def answers_file_name(part: str) -> str:
    """Return the name of a part's answers file, in the directory of its method.

    Args:
        part: ``dev`` or ``test``.

    Returns:
        ``answers_<part>.jsonl``.

    Raises:
        ValueError: If the part is not one of ``PARTS``.
    """
    if part not in PARTS:
        raise ValueError(f"Unknown part {part!r}: expected one of {PARTS}.")
    return f"answers_{part}.jsonl"


def read_answers(path: Path) -> list[LlmAnswer]:
    """Read every answer of an answers file, in the order they were written.

    A last line with no newline is an interrupted write: it is not an answer, even if it parses.
    It is reported on stderr and ignored, so its request is sent again.

    Args:
        path: The answers file.

    Returns:
        The answers; none when the file does not exist.

    Raises:
        ValueError: If a line that ends with a newline is not an answer, or two lines hold
            the same ``request_sha256``.
    """
    path = Path(path)
    if not path.exists():
        return []
    *lines, partial = path.read_bytes().split(_NEWLINE)
    if partial:
        print(
            f"{path}: ignored a last line of {len(partial)} bytes with no newline, left by an "
            "interrupted write: its request will be sent again.",
            file=sys.stderr,
        )
    answers = [_parse_line(path, number, line) for number, line in enumerate(lines, start=1)]
    _check_distinct(path, answers)
    return answers


def append_answer(path: Path, answer: LlmAnswer) -> None:
    """Add one answer to an answers file, as one complete line.

    A partial last line left by an interrupted write is removed first. The line, ending in a
    newline, is then written in one call and flushed, so a killed process leaves it whole or
    absent; a power cut may still cut it short, which ``read_answers`` tolerates. The file and
    its folder are created when missing.

    Args:
        path: The answers file.
        answer: The answer to add.

    Raises:
        ValueError: If the file already has a line for the answer's ``request_sha256``; the
            file is left as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.seek(0)
        stored = file.read()
        complete = stored[: stored.rfind(_NEWLINE) + 1]
        # A request hash is 64 hex digits that no other field of an answer holds.
        if answer.request_sha256.encode() in complete:
            raise ValueError(
                f"{path} already has a line for the request {answer.request_sha256[:12]}: an "
                "answers file keeps one line per request, and never rewrites it."
            )
        if len(complete) < len(stored):
            file.truncate(len(complete))
        file.write(answer.to_json().encode("utf-8") + _NEWLINE)
        file.flush()


def check_append_only(path: Path, versions: Sequence[tuple[str, bytes]]) -> None:
    """Raise unless an answers file was only ever appended to, in git and on disk (docs/adr/0003).

    A line is final once written, failed or not: each committed version of the file holds the
    complete lines of the version before it, then new lines, and so does the file on disk. A
    last line without a newline, an interrupted write, is not a line: ``append_answer``
    replaces it.

    Args:
        path: The answers file; it may not exist yet.
        versions: Its bytes at each commit that changed it, oldest first, by commit sha, as
            ``provenance.committed_versions`` reads them; none when it was never committed.

    Raises:
        ValueError: If a version, or the file on disk, lacks or changes a complete line of the
            version before it.
    """
    path = Path(path)
    on_disk = path.read_bytes() if path.is_file() else b""
    named = [*((f"its commit {sha[:12]}", text) for sha, text in versions)]
    named.append(("the file on disk", on_disk))
    for (earlier, older), (later, newer) in itertools.pairwise(named):
        if not newer.startswith(older[: older.rfind(_NEWLINE) + 1]):
            raise ValueError(
                f"{path}: {later} lacks or changes a line of {earlier}. A line of an answers "
                "file is final once written, failed or not, and the file is only appended to "
                "(docs/adr/0003): restore the lines from git."
            )


def requests_to_send(
    requests: Sequence[LlmRequest], answers: Sequence[LlmAnswer]
) -> list[LlmRequest]:
    """Keep the requests that have no answer yet: any answer, failed or not, is final.

    Args:
        requests: The requests of a part, one per problem.
        answers: The part's answers file, read with ``read_answers``.

    Returns:
        The requests whose hash no answer holds, in order.

    Raises:
        ValueError: If two requests are for one problem, or render the same request.
    """
    hashes = _hashes_by_problem(requests)
    answered = {answer.request_sha256 for answer in answers}
    return [request for request in requests if hashes[request.sample_id] not in answered]


def check_same_requests(requests: Sequence[LlmRequest], answers: Sequence[LlmAnswer]) -> None:
    """Raise unless every answer is to the request that its problem has now (docs/adr/0003).

    A prompted method never changes after its first dev request: an answer whose hash is not
    its problem's current request hash means the prompt, the schema, the fields or an
    answer-shaping setting changed since. It also refuses an answer to a problem that has no
    request.

    Args:
        requests: The requests of a part, one per problem.
        answers: The part's answers file, read with ``read_answers``.

    Raises:
        ValueError: If an answer is not to its problem's current request, or if two requests are
            for one problem or render the same request.
    """
    _check_current(_hashes_by_problem(requests), answers)


def check_complete(requests: Sequence[LlmRequest], answers: Sequence[LlmAnswer]) -> None:
    """Raise unless every request has an answer under its current hash, and no answer is stale.

    Args:
        requests: The requests of a part, one per problem.
        answers: The part's answers file, read with ``read_answers``.

    Raises:
        ValueError: If a request has no answer, or as ``check_same_requests``.
    """
    hashes = _hashes_by_problem(requests)
    _check_current(hashes, answers)
    answered = {answer.request_sha256 for answer in answers}
    pending = [sample_id for sample_id, sha256 in hashes.items() if sha256 not in answered]
    if pending:
        raise ValueError(
            f"There is no answer yet for the problems {problem_list(pending)}: send their "
            "requests with `python -m codeforces_tags.llm run` first."
        )


def _parse_line(path: Path, number: int, line: bytes) -> LlmAnswer:
    """Read one complete line of an answers file as an answer."""
    try:
        return LlmAnswer.from_json(line.decode("utf-8"))
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"{path} line {number} is not an answer ({error}). An answers file is only ever "
            "appended to: restore it from git instead of editing it."
        ) from error


def _check_distinct(path: Path, answers: Sequence[LlmAnswer]) -> None:
    """Raise when two lines of an answers file hold the same request hash."""
    first: dict[str, int] = {}
    for number, answer in enumerate(answers, start=1):
        if answer.request_sha256 in first:
            raise ValueError(
                f"{path} has two lines for the request {answer.request_sha256[:12]}: lines "
                f"{first[answer.request_sha256]} and {number}. An answers file keeps one line per "
                "request hash, which leaves out the problem: two problems whose requests are "
                "identical share one."
            )
        first[answer.request_sha256] = number


def _hashes_by_problem(requests: Sequence[LlmRequest]) -> dict[int, str]:
    """Map each problem to its request's hash, refusing repeated problems and repeated hashes."""
    hashes: dict[int, str] = {}
    for request in requests:
        if request.sample_id in hashes:
            raise ValueError(f"There are two requests for the problem {request.sample_id}.")
        hashes[request.sample_id] = request.sha256()
    counts = Counter(hashes.values())
    twins = [sample_id for sample_id, sha256 in hashes.items() if counts[sha256] > 1]
    if twins:
        raise ValueError(
            f"The problems {problem_list(twins)} render requests that another of them renders too. "
            "An answers file keeps one line per request hash, which leaves out the problem, so "
            "identical requests cannot each be answered: give the prompt a field that tells "
            "the problems apart."
        )
    return hashes


def _check_current(hashes: Mapping[int, str], answers: Sequence[LlmAnswer]) -> None:
    """Raise unless each answer is to the current request of a problem of ``hashes``."""
    unknown = {answer.sample_id for answer in answers if answer.sample_id not in hashes}
    changed = {
        answer.sample_id
        for answer in answers
        if answer.sample_id in hashes and answer.request_sha256 != hashes[answer.sample_id]
    }
    problems = []
    if changed:
        problems.append(
            f"The requests of the problems {problem_list(changed)} are not the ones that were "
            "answered: the prompt, the schema, the fields or a setting that shapes the answer "
            "changed since. A prompted method never changes after its first dev request "
            "(docs/adr/0003): restore the code, or start a new method with a new prompt name."
        )
    if unknown:
        problems.append(
            f"There are answers for the problems {problem_list(unknown)}, which have no request "
            "in this part."
        )
    if problems:
        raise ValueError(" ".join(problems))
