"""Find problems that duplicate each other, so that train/test splits keep them together.

- ``detection``: the grouping to split on (TF-IDF candidates, filtered by judges' verdicts).
- ``audit``: the pipeline that produced the verdicts, with LLM judges and Jinja2 prompts.

The statistics that score a detector against the verdicts, and the pair reports, are read by
notebook 02 only, so they live in ``codeforces_tags.analysis.near_duplicate_quality``.
"""
