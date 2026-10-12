"""The zero-shot methods: the xCodeEval paper's Code2Tag and DesCode2Tag, in two output formats.

Code2Tag asks an LLM for a problem's tags from its solution alone, DesCode2Tag from its
statement, its notes and its solution. Each is asked once for a free-text answer and once for a
strict JSON one, which makes four prompted methods: ``code2tag-text``, ``code2tag-json``,
``descode2tag-text`` and ``descode2tag-json``.

The package holds what defines them: their settings, their approved prompts and their method
specs. It has no prediction code of its own and no ``__main__.py``: the LLM toolkit's commands
serve it, ``python -m codeforces_tags.llm {render,run,score,verify}``. A method never changes
after its first dev request (docs/adr/0003).

Importing this package imports nothing.
"""
