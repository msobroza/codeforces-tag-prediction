"""The F2LLM embedding methods: frozen F2LLM-0.6B embeddings under four tagging methods.

- ``settings``: every setting of the package, and the ``SETTINGS`` its recipes hash.
- ``encoder``: the pinned weights (``fetch`` and the hash checks), the encoder and its device.
- ``embedding``: the encoder inputs, the embedding cache and the feature frame.
- ``estimators``: the embedding blocks, the probe, early fusion and late fusion.
- ``setfit``: SetFit-style contrastive fine-tuning (LoRA) and its estimator.
- ``method``: ``F2LLMMethod``, the tagging method, and the module-level ``load`` the registry names.
- ``analysis``: the shared analysis files, written to ``assets/f2llm/``.
- ``tuning``: the grid studies of the three tuned methods.
- ``drift``: the measure of ``score_atol``, the tolerance of the scores across devices.
- ``__main__``: ``python -m codeforces_tags.methods.f2llm <command>``, with the commands ``fetch``,
  ``embed``, ``reference``, ``tune``, ``export``, ``atol``, ``freeze`` and ``verify``.

Importing this package imports none of them: a method's prediction path loads ``method`` only.
"""
