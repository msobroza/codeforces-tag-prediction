"""Render scikit-learn's diagram of the tag model's pipeline as ``tfidf_estimator.png``.

Notebook 04 shows the picture beside its Mermaid diagrams. It is ``estimator_html_repr`` of
``build_pipeline(params)`` for the frozen ``params.json``: the estimator's own HTML, opened in
headless Chrome with every step box open, and screenshotted at device scale 2. The notebook shows
the PNG instead of the estimator because that HTML weighs 136-170 KB and holds memory addresses
that change on every run.

Run it from the repository root, after a change to ``build_pipeline``, ``ModelParams`` or
``params.json`` (``playwright`` is in the ``figures`` dependency group, which CI does not
install)::

    uv run --group figures python notebooks/figures/estimator_diagram.py

It renders with the system Chrome, or, without one, with the Chromium that
``uv run --group figures playwright install chromium`` downloads. Every http(s) request is
aborted, so the render needs no network and the PNG holds no remote asset. The "Parameters"
tables inside the boxes stay closed: they show the address of every function the pipeline holds.
The PNG is byte-identical across runs on one machine, not necessarily across machines (fonts,
Chrome's version). This is a render tool, like mermaid-cli, not a module the notebooks import.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from playwright.sync_api import Browser, Error, Playwright, Route, sync_playwright
from sklearn.utils import estimator_html_repr

from codeforces_tags.model.training import build_pipeline
from codeforces_tags.tuning.results import load_tuning_results

OUTPUT = Path(__file__).resolve().with_name("tfidf_estimator.png")
DEVICE_SCALE_FACTOR = 2
# The requests to abort. A string glob such as "http*://**" would not intercept them (probed).
REMOTE_REQUEST = re.compile(r"^https?://")
# In scikit-learn's HTML, a step's box opens when its checkbox is checked.
OPEN_EVERY_STEP_BOX = (
    "document.querySelectorAll('input.sk-toggleable__control')"
    ".forEach(box => { box.checked = true; })"
)
# A margin around the diagram, which the screenshot of its element would cut at the border.
MARGIN = "div.sk-container { padding: 12px; }"


def launch(playwright: Playwright) -> Browser:
    """Launch headless Chrome, or else the Chromium that ``playwright install chromium`` fetches.

    Args:
        playwright: The running Playwright driver.

    Returns:
        The browser.
    """
    try:
        return playwright.chromium.launch(channel="chrome")
    except Error:  # no system Chrome
        return playwright.chromium.launch()


def render(html: str, output: Path) -> None:
    """Open an estimator's HTML with every step box open, and screenshot it to ``output``.

    Args:
        html: ``estimator_html_repr`` of an estimator.
        output: The PNG to write.

    Raises:
        RuntimeError: If the page asked for a file over http or https, which the render aborts:
            the picture would lack it.
    """
    aborted: list[str] = []

    def abort(route: Route) -> None:
        aborted.append(route.request.url)
        route.abort()

    with tempfile.TemporaryDirectory() as folder, sync_playwright() as playwright:
        page_file = Path(folder) / "estimator.html"
        page_file.write_text(html, encoding="utf-8")
        browser = launch(playwright)
        page = browser.new_page(color_scheme="light", device_scale_factor=DEVICE_SCALE_FACTOR)
        page.route(REMOTE_REQUEST, abort)
        page.goto(page_file.as_uri())
        if aborted:
            raise RuntimeError(f"The diagram asked for remote files, which it must not: {aborted}")
        page.add_style_tag(content=MARGIN)
        page.evaluate(OPEN_EVERY_STEP_BOX)
        page.locator("div.sk-container").screenshot(path=output)
        browser.close()


def main() -> None:
    """Draw the diagram of the frozen configuration's pipeline into ``tfidf_estimator.png``."""
    pipeline = build_pipeline(load_tuning_results().params)
    render(estimator_html_repr(pipeline), OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
