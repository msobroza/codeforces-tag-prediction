"""Tests for the strict rendering of prompts."""

import jinja2
import pytest

from codeforces_tags.templating import render_string, render_template


def test_render_string_fills_in_the_variables():
    rendered = render_string("Tags: {{ tags | join(', ') }}.", tags=["dp", "math"])

    assert rendered == "Tags: dp, math."


def test_render_string_refuses_a_variable_the_template_needs_and_did_not_get():
    with pytest.raises(jinja2.UndefinedError, match="'code' is undefined"):
        render_string("Here is the code: {{ code }}", problem="x")


def test_render_string_refuses_a_missing_variable_in_a_condition():
    # Jinja's default Undefined is falsy: the branch would be skipped without a word.
    with pytest.raises(jinja2.UndefinedError, match="'flag' is undefined"):
        render_string("{% if flag %}yes{% endif %}")


def test_render_string_refuses_a_missing_attribute():
    with pytest.raises(jinja2.UndefinedError, match="no attribute 'statement'"):
        render_string("{{ problem.statement }}", problem={"id": 1})


def test_render_string_prints_a_none_value_as_none():
    assert render_string("Notes: {{ notes }}", notes=None) == "Notes: None"


@pytest.mark.parametrize(
    "value",
    [
        "{{ x }}",
        "{{ 1 + 1 }}",
        "{% if x %}x{% endif %}",
        "{# a comment #}",
        "}} {{",
        "print('héllo → 日本')",
        "line 1\r\nline 2\r\n",
    ],
)
def test_render_string_prints_a_value_verbatim_whatever_it_holds(value):
    # A solution is data: Jinja-looking text, non-ASCII text and Windows newlines in a value
    # reach the prompt as they are, and are never rendered again.
    assert render_string("[{{ code }}]", code=value) == f"[{value}]"


def test_render_string_does_not_escape_html_characters():
    code = "if a < b and c > 'd' & \"e\": pass"

    assert render_string("{{ code }}", code=code) == code


def test_render_string_drops_the_final_newline_of_a_template():
    # A .j2 file ends with a newline, as any text file does: it is not part of the prompt.
    assert render_string("Provide the tags.\n") == "Provide the tags."
    assert render_string("Provide the tags.\n\n") == "Provide the tags.\n"


def test_render_string_drops_a_leading_comment_and_the_newline_after_it():
    template = "{#- Why this wording. -#}\nPredict the tags of {{ problem }}.\n"

    assert render_string(template, problem="a problem") == "Predict the tags of a problem."


@pytest.mark.parametrize("tag", ['{% include "part.j2" %}', '{% extends "base.j2" %}'])
def test_render_string_has_no_loader_so_a_template_renders_on_its_own(tag):
    # The snapshot of a prompt in a method folder is one string: it cannot include or extend
    # another file, and a prompt template never needs to.
    with pytest.raises(TypeError, match="no loader"):
        render_string(tag)


def test_render_string_renders_the_same_template_with_other_values():
    template = "Code: {{ code }}"

    assert render_string(template, code="a") == "Code: a"
    assert render_string(template, code="b") == "Code: b"
    with pytest.raises(jinja2.UndefinedError):
        render_string(template)


def test_a_variable_may_be_called_name_template_or_template_dir(tmp_path):
    # The three are positional-only parameters, so they are free as variable names; pylint's
    # warning W1117 says so at each call, which is the point of this test.
    (tmp_path / "greeting.j2").write_text("{{ name }} {{ template_dir }}", encoding="utf-8")

    # pylint: disable-next=kwarg-superseded-by-positional-arg
    assert render_string("{{ template }}", template="t") == "t"
    # pylint: disable-next=kwarg-superseded-by-positional-arg
    assert render_template("greeting", tmp_path, name="n", template_dir="d") == "n d"


def test_render_template_reads_the_j2_file_of_that_name_and_follows_includes(tmp_path):
    whole = '{% include "part.j2" %}\n\nwhole {{ n }}\n'
    (tmp_path / "part.j2").write_text("part {{ n }}\n", encoding="utf-8")
    (tmp_path / "whole.j2").write_text(whole, encoding="utf-8")

    assert render_template("whole", tmp_path, n=1) == "part 1\n\nwhole 1"


def test_render_template_refuses_a_variable_the_template_needs_and_did_not_get(tmp_path):
    (tmp_path / "prompt.j2").write_text("Read {{ batch_file }}.\n", encoding="utf-8")

    with pytest.raises(jinja2.UndefinedError, match="'batch_file' is undefined"):
        render_template("prompt", tmp_path)


def test_render_template_refuses_a_template_that_is_not_in_the_folder(tmp_path):
    with pytest.raises(jinja2.TemplateNotFound, match="missing.j2"):
        render_template("missing", tmp_path)


def test_render_template_and_render_string_share_one_environment(tmp_path):
    source = "{#- Why. -#}\nTags: {{ tags | join(', ') }} & {{ code }}\n"
    (tmp_path / "prompt.j2").write_text(source, encoding="utf-8")
    values = {"tags": ["dp", "math"], "code": "a < b"}

    assert render_template("prompt", tmp_path, **values) == render_string(source, **values)
