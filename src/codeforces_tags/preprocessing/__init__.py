"""Turn raw samples into clean, model-ready rows.

The entry point is ``codeforces_tags.preprocessing.pipeline.preprocess``, which runs every
step in order: typed columns (``columns``), text cleaning (``text``) and code parsing
(``source_code``).
"""
