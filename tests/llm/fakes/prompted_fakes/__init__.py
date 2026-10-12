"""Fake prompted packages that the LLM toolkit's commands serve in the tests (``test_main.py``).

- ``zero_shot``: the zero-shot methods ``fake-text`` and ``fake-json``, one per output format,
  sharing the prompt ``fake`` (``tags.j2`` and, for ``json``, ``tags.schema.json``), and a
  ``verify_extra``.
- ``few_shot``: ``fake-few-shot``, a fitted prompted method whose prompt shows, for each problem,
  the fit problems nearest to it (``few_shot.j2``); its ``load`` reads them from the dataset.

The tests import them as the top-level package ``prompted_fakes`` (``monkeypatch.syspath_prepend``
on ``tests/llm/fakes``), never as ``tests.…``: ``import tests`` finds a dependency's installed
``tests`` package. Neither is in ``METHODS``: each test registers them in its own registry.
"""
