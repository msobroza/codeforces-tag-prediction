"""Turn a raw problem into its predicted focus tags: the tag model, the method ``tfidf``.

Entry points, used by notebook 04 and the tuning command:

- ``training``: ``build_pipeline`` (the pipeline of a configuration: ``raw_to_problems``, the
  feature blocks, one linear model per tag), ``cross_fit`` (out-of-fold scores and
  cross-fitted thresholds on the dev set) and ``fit_tag_model`` (the same, then a refit on every
  given problem: the tag model), both through ``CrossFittedTagThresholds``.
- ``tag_model``: ``TagModel``, the score method ``tfidf`` of ``METHODS``, which scores raw
  problems and predicts their focus tags, and the module-level ``load`` of its method folder.
- ``first_look``: ``first_look_pipeline``, notebook 04's untuned reference model.

Building blocks: ``params`` (``ModelParams``, one configuration of the search space),
``features`` (the statement, code and cue feature blocks), ``thresholds`` (the per-tag
thresholds and ``CrossFittedTagThresholds``) and ``platt_lipton`` (the Platt + Lipton
threshold rule of the calibrated Jev twins, docs/adr/0011, and ``PlattLiptonThresholds``,
which runs it inside the cross-fitting of ``CrossFittedTagThresholds``; the tag model never
uses it).

A saved tag model stores types by their import path (``tag_model.STORED_TYPES``): the
functions of ``features.py``, ``raw_to_problems`` and ``CrossFittedTagThresholds``. Renaming
or moving one of them makes saved tag models unloadable.
``CrossFittedTagThresholds`` is never edited, and ``PlattLiptonThresholds`` subclasses it.
Nothing here imports mlflow or optuna.
"""
