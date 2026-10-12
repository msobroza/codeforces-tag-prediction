# Slides

The presentation deck: the task and the data, the protocol (near-duplicates, the split, the
pipeline, how methods are chosen), the results against the xCodeEval paper, how each method works,
the code, the command line and serving, and what we learned.

Its tables compare 13 methods. They leave out the three Jev reference variants that the
[README](../../README.md) lists (`jev-statement-code-gepa`, `jev-statement-code-platt` and
`jev-statement-code-gepa-platt`); the conclusion mentions the GEPA search in one line.
The deck is a snapshot: its counts (13 frozen methods, notebooks 01 to 09, 110 test files,
8 ADRs) date from October 2026; the README gives the current numbers of methods,
notebooks and ADRs.

- [`slides.pdf`](slides.pdf): the 27 slides, 1920×1080.
- [`slides.html`](slides.html): the same slides in one page, each followed by its speaker notes
  on screen; printing it gives `slides.pdf`, without the notes.
- [`source/`](source/): the slides as they were authored: `deck.json` (their order, sections and
  fonts) and one HTML fragment per slide in `source/slides/`, with inline styles and the speaker
  notes in its `<aside>`.
- [`images/`](images/): the two images the slides show: a cropped view of the MLflow trace drawer
  of `notebooks/figures/mlflow/07_trace_drawer.png`, and the Streamlit app of notebook 12
  (`notebooks/figures/streamlit/new_problem.png`).

The numbers on the slides come from the frozen files under `assets/` and from the notebooks; no
command regenerates the deck. To render the PDF again after editing
`slides.html`, print it with headless Chrome (the page sets its own 1920×1080 page size):

```bash
google-chrome --headless=new --no-pdf-header-footer --virtual-time-budget=10000 --print-to-pdf=docs/slides/slides.pdf docs/slides/slides.html
```
