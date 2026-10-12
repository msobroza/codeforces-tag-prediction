"""Tests for the answers files: reading, appending, resuming, and the gates on their requests."""

import pytest

from codeforces_tags.config import METHODS_ASSETS_DIR
from codeforces_tags.llm.store import (
    answers_file,
    answers_file_name,
    append_answer,
    check_append_only,
    check_complete,
    check_same_requests,
    read_answers,
    requests_to_send,
)


def answer_to(make_answer, request, **changes):
    """Return the answer to a request: the same problem, under the request's hash."""
    return make_answer(sample_id=request.sample_id, request_sha256=request.sha256(), **changes)


def line_of(answer):
    """Return the bytes that an answers file holds for an answer: its line and a newline."""
    return answer.to_json().encode("utf-8") + b"\n"


def test_answers_file_is_the_answers_of_a_part_in_the_method_directory():
    assert (
        answers_file("zero_shot", "dev") == METHODS_ASSETS_DIR / "zero_shot" / "answers_dev.jsonl"
    )
    assert answers_file("a-1", "test").name == "answers_test.jsonl"


@pytest.mark.parametrize("part", ["train", "", "../dev", "Dev"])
def test_answers_file_refuses_a_part_that_is_not_dev_or_test(part):
    with pytest.raises(ValueError, match="Unknown part"):
        answers_file("zero_shot", part)


def test_answers_file_refuses_a_name_that_is_not_a_method_name():
    with pytest.raises(ValueError, match="is not a method name"):
        answers_file("../zero_shot", "dev")


def test_answers_file_name_names_the_answers_file_of_a_part():
    assert answers_file_name("test") == "answers_test.jsonl"
    with pytest.raises(ValueError, match="Unknown part"):
        answers_file_name("train")


def test_read_answers_of_a_file_that_is_not_there_is_empty(tmp_path):
    assert read_answers(tmp_path / "answers_dev.jsonl") == []


def test_read_answers_reads_the_lines_in_the_order_they_were_written(make_answer, tmp_path):
    answers = [
        make_answer(sample_id=3, request_sha256="3" * 64),
        make_answer(sample_id=1, request_sha256="1" * 64, tags=("dp",)),
    ]
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(b"".join(line_of(answer) for answer in answers))

    assert read_answers(path) == answers


def test_read_answers_reports_nothing_for_a_file_that_ends_with_a_newline(
    make_answer, tmp_path, capsys
):
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(make_answer()))

    read_answers(path)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("bad_line", "reason"),
    [
        pytest.param(b"not json", "Expecting value", id="not JSON"),
        pytest.param(b"", "Expecting value", id="a blank line"),
        pytest.param(b"[1, 2]", "JSON object", id="not an object"),
        pytest.param(b'{"sample_id": 1}', "missing", id="fields missing"),
        pytest.param(b"\xff\xfe", "utf-8", id="not UTF-8"),
    ],
)
def test_read_answers_refuses_a_line_that_is_not_an_answer(make_answer, tmp_path, bad_line, reason):
    first = make_answer(sample_id=1, request_sha256="1" * 64)
    third = make_answer(sample_id=3, request_sha256="3" * 64)
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(first) + bad_line + b"\n" + line_of(third))

    with pytest.raises(ValueError, match=rf"answers_dev.jsonl line 2 is not an answer .*{reason}"):
        read_answers(path)


def test_read_answers_refuses_a_bad_last_line_that_ends_with_a_newline(make_answer, tmp_path):
    # Only a last line with no newline is an interrupted write: with one, the file is corrupt.
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(make_answer()) + b'{"sample_id": 2}\n')

    with pytest.raises(ValueError, match="line 2 is not an answer"):
        read_answers(path)


def test_read_answers_refuses_two_lines_for_one_request(make_answer, tmp_path):
    # Two problems whose requests are identical share a request hash.
    twins = [make_answer(sample_id=5), make_answer(sample_id=9)]
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(b"".join(line_of(answer) for answer in twins))

    with pytest.raises(ValueError, match=r"two lines for the request aaaaaaaaaaaa: lines 1 and 2"):
        read_answers(path)


def test_read_answers_ignores_and_reports_a_last_line_with_no_newline(
    make_answer, tmp_path, capsys
):
    done = make_answer(sample_id=1, request_sha256="1" * 64)
    partial = b'{"request_sha256": "22222'
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(done) + partial)

    assert read_answers(path) == [done]
    report = capsys.readouterr().err
    assert f"answers_dev.jsonl: ignored a last line of {len(partial)} bytes" in report
    assert "no newline" in report
    assert "interrupted write" in report


def test_read_answers_ignores_a_last_line_with_no_newline_even_when_it_parses(
    make_answer, tmp_path, capsys
):
    # The write that would have ended the line never finished: its request is sent again.
    done = make_answer(sample_id=1, request_sha256="1" * 64)
    whole_but_unterminated = make_answer(sample_id=2, request_sha256="2" * 64)
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(done) + whole_but_unterminated.to_json().encode("utf-8"))

    assert read_answers(path) == [done]
    assert "ignored a last line" in capsys.readouterr().err


def test_read_answers_ignores_a_last_line_cut_inside_a_character(make_answer, tmp_path):
    done = make_answer(sample_id=1, request_sha256="1" * 64)
    cut = make_answer(sample_id=2, output_text="dp → math").to_json().encode("utf-8")
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(done) + cut[: cut.index("→".encode()) + 1])

    assert read_answers(path) == [done]


def test_read_answers_of_a_file_holding_only_a_partial_line_is_empty(make_answer, tmp_path):
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(make_answer())[:40])

    assert read_answers(path) == []


def test_append_answer_creates_the_file_and_its_folder_and_writes_one_line_per_answer(
    make_answer, tmp_path
):
    path = tmp_path / "assets" / "methods" / "fake" / "answers_dev.jsonl"
    first = make_answer(sample_id=1, request_sha256="1" * 64)
    second = make_answer(sample_id=2, request_sha256="2" * 64, output_text="trees, dp → math")

    append_answer(path, first)
    append_answer(path, second)

    assert path.read_bytes() == line_of(first) + line_of(second)
    assert read_answers(path) == [first, second]


def test_append_answer_refuses_a_second_line_for_a_request_and_leaves_the_file_as_it_was(
    make_answer, tmp_path
):
    path = tmp_path / "answers_dev.jsonl"
    append_answer(path, make_answer(sample_id=1))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="already has a line for the request aaaaaaaaaaaa"):
        append_answer(path, make_answer(sample_id=2))

    assert path.read_bytes() == before


def test_an_interrupted_write_is_reported_then_replaced_by_the_next_append(
    make_answer, tmp_path, capsys
):
    path = tmp_path / "answers_dev.jsonl"
    first = make_answer(sample_id=1, request_sha256="1" * 64)
    second = make_answer(sample_id=2, request_sha256="2" * 64)
    third = make_answer(sample_id=3, request_sha256="3" * 64, output_text="dp → math")
    append_answer(path, first)
    append_answer(path, second)
    # The process dies in the middle of the third write, and cuts a character in two.
    whole = third.to_json().encode("utf-8")
    with path.open("ab") as file:
        file.write(whole[: whole.index("→".encode()) + 1])

    assert read_answers(path) == [first, second]
    assert "interrupted write" in capsys.readouterr().err

    append_answer(path, third)

    assert read_answers(path) == [first, second, third]
    assert path.read_bytes() == line_of(first) + line_of(second) + line_of(third)
    assert capsys.readouterr().err == ""


def test_an_interrupted_first_write_leaves_an_empty_file_for_the_next_append(make_answer, tmp_path):
    path = tmp_path / "answers_dev.jsonl"
    answer = make_answer()
    path.write_bytes(line_of(answer)[:30])

    append_answer(path, answer)

    assert path.read_bytes() == line_of(answer)


def test_append_answer_accepts_the_answer_whose_interrupted_line_it_replaces(make_answer, tmp_path):
    # The cut line already holds the request's hash: it is dropped before the hash is looked for.
    path = tmp_path / "answers_dev.jsonl"
    answer = make_answer()
    whole = answer.to_json().encode("utf-8")
    path.write_bytes(whole[: whole.index(answer.request_sha256.encode()) + 64])

    append_answer(path, answer)

    assert path.read_bytes() == line_of(answer)


def test_requests_to_send_keeps_the_requests_that_have_no_line_in_their_order(
    make_answer, requests_for
):
    requests = requests_for(4)
    answers = [answer_to(make_answer, requests[2]), answer_to(make_answer, requests[0])]

    assert requests_to_send(requests, answers) == [requests[1], requests[3]]


def test_requests_to_send_sends_everything_when_nothing_is_answered(requests_for):
    requests = requests_for(3)

    assert requests_to_send(requests, []) == requests


@pytest.mark.parametrize(
    ("status", "output_text"),
    [
        ("completed", "dp"),
        ("incomplete", "dp, ma"),
        ("refused", ""),
        ("api_error:400", None),
    ],
)
def test_requests_to_send_counts_every_answer_as_final_whatever_its_status(
    make_answer, requests_for, status, output_text
):
    [request] = requests_for(1)
    answer = answer_to(make_answer, request, status=status, output_text=output_text)

    assert requests_to_send([request], [answer]) == []


def test_requests_to_send_refuses_requests_that_render_alike(make_request):
    # The hash leaves out the problem: two identical requests could not each be answered.
    twins = [make_request(sample_id=4), make_request(sample_id=8)]

    with pytest.raises(ValueError, match=r"The problems \[4, 8\] render requests that another"):
        requests_to_send(twins, [])


def test_requests_to_send_refuses_two_requests_for_one_problem(make_request):
    requests = [make_request(sample_id=4, input="a"), make_request(sample_id=4, input="b")]

    with pytest.raises(ValueError, match="two requests for the problem 4"):
        requests_to_send(requests, [])


def test_check_same_requests_accepts_answers_to_the_current_requests(make_answer, requests_for):
    requests = requests_for(3)
    answers = [answer_to(make_answer, request) for request in requests[:2]]

    check_same_requests(requests, answers)
    check_same_requests(requests, [])


def test_check_same_requests_names_the_problems_whose_request_changed(
    make_answer, make_request, requests_for
):
    # The answers of problems 1 to 3 were recorded under their old requests; problem 2's prompt
    # has changed since.
    old = requests_for(3)
    answers = [answer_to(make_answer, request) for request in old]
    new = [old[0], make_request(sample_id=2, input="A new prompt."), old[2]]

    with pytest.raises(
        ValueError, match=r"requests of the problems \[2\] are not the ones"
    ) as error:
        check_same_requests(new, answers)

    assert "new method with a new prompt name" in str(error.value)


def test_check_same_requests_refuses_an_answer_to_a_problem_without_a_request(
    make_answer, requests_for
):
    requests = requests_for(2)
    answers = [answer_to(make_answer, request) for request in requests_for(3)]

    with pytest.raises(ValueError, match=r"answers for the problems \[3\], which have no request"):
        check_same_requests(requests, answers)


def test_check_same_requests_names_both_kinds_of_stale_answer(
    make_answer, make_request, requests_for
):
    answers = [answer_to(make_answer, request) for request in requests_for(3)]
    requests = [make_request(sample_id=1, input="A new prompt."), requests_for(3)[1]]

    with pytest.raises(ValueError) as error:
        check_same_requests(requests, answers)

    assert "problems [1] are not the ones" in str(error.value)
    assert "problems [3], which have no request" in str(error.value)


def test_check_same_requests_refuses_requests_that_render_alike(make_request):
    twins = [make_request(sample_id=4), make_request(sample_id=8)]

    with pytest.raises(ValueError, match="render requests that another"):
        check_same_requests(twins, [])


def test_check_complete_accepts_an_answer_to_every_current_request(make_answer, requests_for):
    requests = requests_for(3)

    check_complete(requests, [answer_to(make_answer, request) for request in requests])


def test_check_complete_names_the_problems_that_have_no_answer_yet(make_answer, requests_for):
    requests = requests_for(5)
    answers = [answer_to(make_answer, requests[index]) for index in (0, 2, 3)]

    with pytest.raises(ValueError, match=r"no answer yet for the problems \[2, 5\]"):
        check_complete(requests, answers)


def test_check_complete_refuses_an_answer_to_an_old_request_for_a_problem(
    make_answer, make_request, requests_for
):
    old = requests_for(2)
    answers = [answer_to(make_answer, request) for request in old]
    new = [old[0], make_request(sample_id=2, input="A new prompt.")]

    with pytest.raises(ValueError, match="are not the ones that were answered"):
        check_complete(new, answers)


def test_check_complete_refuses_an_answer_to_a_problem_without_a_request(make_answer, requests_for):
    answers = [answer_to(make_answer, request) for request in requests_for(3)]

    with pytest.raises(ValueError, match="which have no request"):
        check_complete(requests_for(2), answers)


def commit(number):
    """Return the sha of a made-up commit: its one-digit number, 40 times."""
    return str(number) * 40


def test_check_append_only_accepts_versions_that_each_add_lines(tmp_path):
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(b"a\nb\nc\n")

    check_append_only(path, [(commit(1), b"a\n"), (commit(2), b"a\nb\n"), (commit(3), b"a\nb\n")])
    check_append_only(tmp_path / "answers_test.jsonl", [])  # never committed, not written yet


def test_check_append_only_lets_a_version_replace_an_interrupted_last_line(tmp_path):
    # A last line without a newline is an interrupted write, which append_answer replaces.
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(b"a\nc\n")

    check_append_only(path, [(commit(1), b"a\nb")])


@pytest.mark.parametrize(
    ("on_disk", "versions", "message"),
    [
        (
            b"a\nb\n",
            [(commit(1), b"a\nb\n"), (commit(2), b"a\n")],
            "its commit 222222222222 lacks or changes a line of its commit 111111111111",
        ),
        (
            b"a\nB\nc\n",
            [(commit(1), b"a\nb\n")],
            "the file on disk lacks or changes a line of its commit 111111111111",
        ),
        (None, [(commit(1), b"a\n")], "the file on disk lacks or changes a line of its commit"),
    ],
)
def test_check_append_only_refuses_a_removed_or_edited_line(tmp_path, on_disk, versions, message):
    path = tmp_path / "answers_dev.jsonl"
    if on_disk is not None:
        path.write_bytes(on_disk)

    with pytest.raises(ValueError, match=f"answers_dev.jsonl: {message}") as error:
        check_append_only(path, versions)

    assert "final once written, failed or not" in str(error.value)
