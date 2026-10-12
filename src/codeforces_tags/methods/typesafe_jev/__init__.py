"""The TypeSafe Jev methods: Jev is asked one question per focus tag, its probabilities are scores.

- ``settings``: the methods' names (two frozen, one searched, two calibrated twins), their
  studies, each twin's base and the ``TEST_INFORMED`` disclosure they record, the model, the
  seed words (read from the versioned prompt file ``templates/jev-tags.v1.yaml``), the search's
  settings and the transport settings.
- ``words``: Jev's words as data (``JevWords``, ``SEED_WORDS``), the pieces a search rewrites,
  their length caps, the dataset-text check, the words and settings each method hashes, and
  ``base_of``, a twin's base.
- ``questions``: each problem's state, the eight Noul questions and the hashed request.
- ``client``: the only module that imports the TypeSafe SDK, and the mapping of its outcomes onto
  the lines of an answers file.
- ``answers``: the probabilities, the summary and the served-model check of a part's answers.
- ``runner``: sends the pending requests concurrently under a cost cap.
- ``method``: ``JevTagger``, the tagging method, and the module-level ``load`` the registry names.
- ``tracking``: the prompt ``jev-tags`` in MLflow's prompt registry, one version per prompt
  file, registered by ``run`` and linked by ``freeze`` to the logged model and run of each
  method that reads it, the two frozen ones.
- ``objective``: the search's objective, the balanced Brier score of Jev's probabilities
  (``problem_scores``, ``balanced_brier``).
- ``sampling``: ``StratifiedSampler``, the search's minibatches of learning-fold problems, with
  positives of every focus tag in each.
- ``reflection``: what the reflection model reads and answers (the records of an evaluated
  minibatch, ``reflection_request``, ``read_proposal``), and ``Reflector``, its cached call.
- ``jev_cache``: ``JevCache``, the search's Jev answers, one answers file per set of candidate
  words, the seed's filled from the committed answers.
- ``tuning``: the search over Jev's words: GEPA's adapter, its stoppers, the recorder of its
  history and candidates, and ``run_search``; the only module here that imports gepa.
- ``search_record``: the frozen record of a search in the searched method's ``tuning/``
  (``SearchRecord``, ``read_search_record``), the files ``export`` writes once, the selection
  rule and the run record's search entries; it imports neither gepa nor mlflow.
- ``export``: ``export_search``, which freezes a finished search's record once, after the
  checks of docs/adr/0010; it imports neither gepa nor mlflow.
- ``search_tracking``: the search in MLflow, logged live by ``tune`` (``LiveSearchLog``) or
  replayed from ``tuning/`` (``log_search``), into the experiment ``jev-gepa``; it imports mlflow
  only once ``use_store`` has set MLflow's environment.
- ``search_commands``: the handlers of ``tune``, ``export`` and ``log-search``, which
  ``__main__`` dispatches to once their gates pass.
- ``platt``: the threshold rules R0 to R2 of a calibrated twin (docs/adr/0011), the tables
  of its dev-only record in ``tuning/`` (``rules.csv``, ``nested.csv``, ``stability.csv``)
  and ``export_platt``, which writes them once; it imports neither gepa nor mlflow.
- ``platt_commands``: the handlers of a calibrated twin's commands, which ``__main__``
  dispatches to once their gates pass, and the git gates on its base's files.
- ``templates/``: ``jev-tags.v1.yaml``, the frozen methods' words; ``gepa/``, the reflection
  prompts and strict answer schemas of a tag piece and of the convention, rendered by
  ``reflection``.
- ``__main__``: ``python -m codeforces_tags.methods.typesafe_jev
  render|run|freeze|verify|tune|export|log-search``; for a calibrated twin, ``export`` writes
  its dev-only record, ``freeze`` and ``verify`` treat it as a composite over its base's
  files, and ``run`` and ``render`` refuse it.

Importing this package imports none of them: a method's prediction path loads ``method`` only.
"""
