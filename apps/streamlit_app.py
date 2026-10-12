"""Streamlit client of a model server: one problem's focus tags, asked live (docs/adr/0012).

Run it from the repository root, where Streamlit reads ``.streamlit/config.toml``, once a model
server answers (``docker compose -f deploy/docker-compose.yml``, ``docker run``, a
``kubectl port-forward`` or ``serving.local``):

    uv run --group app streamlit run apps/streamlit_app.py

It takes no option. It asks the model server of ``$CODEFORCES_SERVING_URL``
(http://127.0.0.1:8080 by default), reads the test-set problems from ``$CODEFORCES_DATA_DIR``,
and links the served tagging method to the ``tag-methods`` experiment, its run and its prompt
versions in the MLflow UI of ``$CODEFORCES_MLFLOW_UI_URL`` (http://127.0.0.1:5000 by default),
which ``uv run python -m codeforces_tags.serving mlflow-ui`` serves; both URLs can be edited in
the sidebar. The script holds layout only: the request and its errors, the form's rules, the
tables and the links come from the ``client``, ``problems`` and ``mlflow_links`` modules of
``codeforces_tags.serving``, and it imports nothing of mlflow.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from itertools import zip_longest
from urllib.parse import urlsplit

import altair as alt
import pandas as pd
import streamlit as st

from codeforces_tags.cli import CliError
from codeforces_tags.config import (
    CLI_JSON_LIST_FIELDS,
    CLI_REQUIRED_FIELDS,
    FOCUS_TAGS,
    MLFLOW_METHODS_EXPERIMENT,
    SERVING_DEFAULT_URL,
    SERVING_MLFLOW_UI_URL,
)
from codeforces_tags.preprocessing.columns import clean_columns
from codeforces_tags.serving.client import (
    ServedPrediction,
    ServingError,
    ping,
    predict,
    server_url,
)
from codeforces_tags.serving.mlflow_links import (
    UI_START_COMMAND,
    method_links,
    mlflow_ui_url,
    ui_alive,
)
from codeforces_tags.serving.problems import (
    FORM_FIELDS,
    FrozenRow,
    display_text,
    frozen_prediction,
    load_test_set,
    problem_from_form,
    tag_table,
)

TEST_MODE = "Test-set problem"
NEW_MODE = "New problem"
ANY_TAG = "any focus tag"
# The command the sidebar shows when no model server answers: tfidf's image, the smallest, served
# on the CPU once `python -m codeforces_tags.serving build --method tfidf` has made it.
SERVER_START_COMMAND = (
    "METHOD=tfidf docker compose -f deploy/docker-compose.yml --profile cpu up --wait"
)
# The raw fields the test-set view shows: the statement's three parts, the notes, the solution.
STATEMENT_COLUMNS = ("prob_desc_description", "prob_desc_input_spec", "prob_desc_output_spec")
NOTES_COLUMN = "prob_desc_notes"
SOLUTION_COLUMN = "source_code"
# The data editor's columns: one row per pair of a sample input and its sample output.
SAMPLE_COLUMNS = ("sample input", "sample output")


def main() -> None:
    """Lay out the page: the model server in the sidebar, a problem, its prediction, the links."""
    st.set_page_config(page_title="Codeforces tags", layout="wide")
    st.title("Focus tags from a model server")
    url = server_sidebar()
    mode = st.radio("Problem", (TEST_MODE, NEW_MODE), horizontal=True)
    if mode == TEST_MODE:
        test_set_view(url)
    else:
        new_problem_view(url)
    tracking_sidebar(st.session_state.get("served", {}).get(url))


def server_sidebar() -> str:
    """Show the model server's URL and whether it answers ``/ping``.

    Returns:
        The model server's URL, as typed in the sidebar (``server_url()`` when left blank).
    """
    with st.sidebar:
        st.header("Model server")
        default = server_url()
        url = checked_url(st.text_input("URL", value=default, key="server url"), default)
        if url is None:
            st.error(f"The model server's URL must be an http(s) URL, e.g. {SERVING_DEFAULT_URL}.")
            st.stop()
        if ping(url):
            st.success("It answers /ping.")
        else:
            st.error(
                "It does not answer /ping: no model server runs there, or it is still loading "
                "its tagging method. To start one, build tfidf's image with `uv run python -m "
                "codeforces_tags.serving build --method tfidf`, then serve it:"
            )
            st.code(SERVER_START_COMMAND, language="bash")
    return url


def tracking_sidebar(method: str | None) -> None:
    """Show the served tagging method, and link it to what MLflow tracked about it.

    The links open the MLflow UI: the ``tag-methods`` experiment, the method's run (its dev
    out-of-fold metrics and frozen latencies, and its logged model, which holds the run record's
    parameters and the test metrics), for a prompted or Jev method each prompt version its
    committed run record pins, then those linked to its run in the store (Jev's ``jev-tags``),
    and the prompt registry.

    Args:
        method: The tagging method of the model server's last prediction at this URL; None
            before one.
    """
    with st.sidebar:
        st.caption(f"Served tagging method: {method or 'known once the model server answers'}")
        st.header("MLflow tracking")
        default = mlflow_ui_url()
        ui = checked_url(st.text_input("MLflow UI", value=default, key="mlflow ui"), default)
        if ui is None:
            st.error(f"The MLflow UI's URL must be an http(s) URL, e.g. {SERVING_MLFLOW_UI_URL}.")
            return
        if not ui_alive(ui):
            st.warning(f"The MLflow UI does not answer at {ui}. Start it with:")
            st.code(UI_START_COMMAND, language="bash")
            return
        st.success("It answers /health.")
        if method is None:
            st.caption("Predict once: the links follow the tagging method the server serves.")
            return
        # ValueError: the served name is not a method name, or its run record pins a prompt
        # that is not a version; the model server is any URL the user types.
        try:
            links = method_links(ui, method)
        except (ServingError, ValueError) as error:
            st.error(f"No MLflow links: {error}")
            return
        link(f"Experiment {MLFLOW_METHODS_EXPERIMENT}", links.experiment)
        link(f"Run of {method}", links.run)
        for label, url in links.prompts:
            link(f"Prompt {label}", url)
        link("Prompt registry", links.registry)


def checked_url(typed: str, default: str) -> str | None:
    """Read a URL typed in the sidebar.

    Args:
        typed: The text of the input.
        default: The URL to use when the input is blank.

    Returns:
        The URL; None when it is not an http:// or https:// URL with a host, or cannot be
        parsed at all.
    """
    url = typed.strip() or default
    try:
        parts = urlsplit(url)
    except ValueError:  # an unclosed IPv6 bracket, e.g. http://[::1:8080
        return None
    return url if parts.scheme in ("http", "https") and parts.netloc else None


def link(label: str, url: str | None) -> None:
    """Show a link button to the MLflow UI, or say that the store lacks the entry.

    Args:
        label: What the link opens.
        url: Its URL; None when the store holds no such entry.
    """
    if url is None:
        st.caption(f"{label}: not logged in this store")
    else:
        st.link_button(label, url, width="stretch")


@st.cache_resource(show_spinner="Reading the test set")
def cached_test_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the test-set problems and their true focus tags, once per Streamlit server.

    ``st.cache_resource`` keeps the frames themselves, which the app never changes:
    ``st.cache_data`` would pickle them and unpickle a copy on every rerun (CLAUDE.md, No
    pickle).

    Returns:
        The test-set problems and their true focus tags, as ``load_test_set`` gives them.
    """
    return load_test_set()


def test_set_view(url: str) -> None:
    """Pick a test-set problem, show it, and ask its focus tags beside the frozen ones.

    Args:
        url: The model server's URL.
    """
    st.caption(
        "A demonstration on frozen test-set problems: it computes no metric and chooses "
        "nothing (ADR 0001); every tagging method was chosen on the dev set."
    )
    try:
        problems, truth = cached_test_set()
    except (CliError, OSError, ValueError) as error:
        st.error(f"No test set: {error}")
        return
    tag = st.selectbox("True focus tag", (ANY_TAG, *FOCUS_TAGS))
    ids = truth.index if tag == ANY_TAG else truth.index[truth[tag] == 1]
    sample_id = st.selectbox(
        f"Problem ({len(ids)} of the {len(truth)} test-set problems)",
        ids,
        format_func=lambda sample_id: problem_label(problems, sample_id),
    )
    if sample_id is None:
        st.info(f"No test-set problem has the focus tag {tag}.")
        return
    problem = problems.loc[[sample_id]]
    left, right = st.columns((3, 2), gap="large")
    with left:
        show_problem(problem)
    with right:
        if st.button("Predict", type="primary", key="predict test"):
            prediction = ask(url, problem)
            if prediction is not None:
                frozen = frozen_row(prediction.method, int(sample_id))
                prediction_panel(prediction, truth.loc[sample_id].to_dict(), frozen)


def problem_label(problems: pd.DataFrame, sample_id: int) -> str:
    """Name a test-set problem in the picker.

    Args:
        problems: The test-set problems, indexed by ``sample_id``.
        sample_id: The problem's sample id.

    Returns:
        Its sample id and the first words of its description.
    """
    words = " ".join(display_text(problems.at[sample_id, "prob_desc_description"]).split())
    return f"{sample_id}: {words[:70]}"


def show_problem(problem: pd.DataFrame) -> None:
    """Show a problem's statement, notes, sample inputs and outputs, and solution.

    Args:
        problem: One raw problem, indexed by ``sample_id``.
    """
    row = clean_columns(problem).iloc[0]
    statement, notes, samples, solution = st.tabs(("Statement", "Notes", "Samples", "Solution"))
    with statement:
        for column in STATEMENT_COLUMNS:
            st.markdown(f"**{FORM_FIELDS[column].capitalize()}**")
            st.markdown(display_text(row[column]))
    with notes:
        st.markdown(display_text(row[NOTES_COLUMN]) or "No notes.")
    with samples:
        inputs, outputs = (listed(row[field]) for field in CLI_JSON_LIST_FIELDS)
        show_samples(inputs, outputs)
    with solution:
        st.code(row[SOLUTION_COLUMN], language="python")


def show_samples(inputs: list[str], outputs: list[str]) -> None:
    """Show each sample input beside its sample output.

    Args:
        inputs: The sample inputs.
        outputs: The sample outputs, in the same order.
    """
    if not inputs and not outputs:
        st.markdown("No sample input or sample output.")
    for number, (sample_input, sample_output) in enumerate(
        zip_longest(inputs, outputs, fillvalue=""), start=1
    ):
        left, right = st.columns(2)
        left.caption(f"Sample input {number}")
        left.code(sample_input, language=None)
        right.caption(f"Sample output {number}")
        right.code(sample_output, language=None)


def listed(value: object) -> list[str]:
    """Return a sample field's list of texts, or an empty list for a null field."""
    return value if isinstance(value, list) else []


def new_problem_view(url: str) -> None:
    """Type a new problem and ask its focus tags.

    Args:
        url: The model server's URL.
    """
    left, right = st.columns((3, 2), gap="large")
    with left:
        texts = {column: form_text(column) for column in FORM_FIELDS if column != SOLUTION_COLUMN}
        pairs = sample_pairs()
        texts[SOLUTION_COLUMN] = form_text(SOLUTION_COLUMN)
    with right:
        if st.button("Predict", type="primary", key="predict new"):
            try:
                problem = problem_from_form(texts, pairs)
            except ValueError as error:
                st.error(f"No prediction: {error}")
                return
            prediction = ask(url, problem)
            if prediction is not None:
                prediction_panel(prediction)


def form_text(column: str) -> str:
    """Show the text area of one field of the form.

    Args:
        column: The raw field, a key of ``FORM_FIELDS``.

    Returns:
        The typed text, as typed.
    """
    label = FORM_FIELDS[column].capitalize()
    if column in CLI_REQUIRED_FIELDS:
        label = f"{label} (required)"
    height = 250 if column == SOLUTION_COLUMN else 120
    return st.text_area(label, key=f"form {column}", height=height)


def sample_pairs() -> list[tuple[str, str]]:
    """Show the data editor of the sample inputs and outputs, one row per pair.

    Returns:
        Each row's sample input and sample output, a cell left empty read as "".
    """
    edited = st.data_editor(
        pd.DataFrame({column: [""] for column in SAMPLE_COLUMNS}),
        num_rows="dynamic",
        hide_index=True,
        key="form samples",
        column_config={
            column: st.column_config.TextColumn(column.capitalize(), width="large")
            for column in SAMPLE_COLUMNS
        },
    )
    return list(edited.fillna("").itertuples(index=False, name=None))


def ask(url: str, problem: pd.DataFrame) -> ServedPrediction | None:
    """Send one problem to the model server, or say in one message why it predicted nothing.

    Args:
        url: The model server's URL.
        problem: One raw problem, indexed by ``sample_id``.

    Returns:
        The model server's prediction, or None once its failure is shown.
    """
    try:
        with st.spinner("Asking the model server"):
            [prediction] = predict(url, problem)
    except (ServingError, CliError) as error:  # one line each, as client.py and cli word them
        st.error(f"No prediction: {error}")
        return None
    st.session_state.setdefault("served", {})[url] = prediction.method
    return prediction


def frozen_row(method: str, sample_id: int) -> FrozenRow | None:
    """Read the served method's frozen test-set prediction of a problem, or say it has none.

    Args:
        method: The served tagging method.
        sample_id: The test-set problem's sample id.

    Returns:
        The frozen prediction, or None when this checkout holds none or it cannot be read.
    """
    try:
        frozen = frozen_prediction(method, sample_id)
    except (OSError, ValueError) as error:
        st.warning(f"No frozen test-set prediction of {method}: {error}")
        return None
    if frozen is None:
        st.info(f"This checkout holds no frozen test-set prediction of {method} for this problem.")
    return frozen


def prediction_panel(
    prediction: ServedPrediction,
    truth: Mapping[str, int] | None = None,
    frozen: FrozenRow | None = None,
) -> None:
    """Show the model server's prediction for one problem.

    Args:
        prediction: The prediction.
        truth: The problem's true focus tags, for a test-set problem.
        frozen: The served method's frozen prediction of the problem, for a test-set problem.
    """
    st.subheader(f"Prediction of {prediction.method}")
    if prediction.failed:
        st.warning(f"The API's answer failed: no tag. Its request cost {usd(prediction.cost_usd)}.")
    st.markdown(f"**Predicted focus tags:** {tag_list(prediction.tags)}")
    if truth is not None:
        st.markdown(f"**True focus tags:** {tag_list(tag for tag in FOCUS_TAGS if truth[tag])}")
    if frozen is not None:
        frozen_tags = tag_list(tag for tag in FOCUS_TAGS if frozen.tags[tag])
        st.markdown(f"**Frozen prediction:** {frozen_tags}")
    latency, cost = st.columns(2)
    latency.metric("Round trip", f"{prediction.seconds:.2f} s")
    if prediction.cost_usd is not None:
        cost.metric("Cost", usd(prediction.cost_usd))
    st.dataframe(tag_table(prediction, truth, frozen).dropna(axis="columns", how="all"))
    # A model server at any typed URL may answer only some of the score columns: no chart then.
    if all(tag in prediction.scores for tag in FOCUS_TAGS):
        st.altair_chart(score_chart(prediction))
        st.caption(
            "Each bar is a focus tag's score and the black tick its per-tag threshold, chosen "
            "on the dev set: the tag is predicted when its score reaches the tick. A tag "
            "without a tick is never predicted."
        )


def tag_list(tags: Iterable[str]) -> str:
    """Join focus tags for display, or say that there is none."""
    return ", ".join(tags) or "none"


def usd(cost: float | None) -> str:
    """Write a request's cost in US dollars."""
    return "unknown" if cost is None else f"${cost:.6f}"


def score_chart(prediction: ServedPrediction) -> alt.LayerChart:
    """Draw each focus tag's score as a bar, with a tick at its per-tag threshold.

    A tag whose per-tag threshold is infinite (it is never predicted) gets no tick.

    Args:
        prediction: A prediction of a score method.

    Returns:
        The bars and the ticks, one row per focus tag.
    """
    scores = pd.DataFrame(
        {
            "focus tag": FOCUS_TAGS,
            "score": [prediction.scores[tag] for tag in FOCUS_TAGS],
            "predicted": ["yes" if tag in prediction.tags else "no" for tag in FOCUS_TAGS],
        }
    )
    thresholds = pd.DataFrame(
        [(tag, value) for tag, value in prediction.thresholds.items() if math.isfinite(value)],
        columns=["focus tag", "per-tag threshold"],
    )
    tag_axis = alt.Y("focus tag:N", sort=list(FOCUS_TAGS), title=None)
    bars = (
        alt.Chart(scores)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="score"),
            y=tag_axis,
            color=alt.Color("predicted:N", sort=["yes", "no"], title="predicted"),
            tooltip=["focus tag", "score", "predicted"],
        )
    )
    ticks = (
        alt.Chart(thresholds)
        .mark_tick(color="black", thickness=3)
        .encode(x="per-tag threshold:Q", y=tag_axis, tooltip=["focus tag", "per-tag threshold"])
    )
    return bars + ticks


if __name__ == "__main__":
    main()
