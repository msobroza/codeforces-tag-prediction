"""Every tagging method in MLflow: its run, its logged model and its evaluations (docs/adr/0002).

With ``tuning/``, ``llm/__main__.py``, each method's ``__main__.py``, ``tracking.py`` and
``tuning.py``, the TypeSafe search's ``search_tracking.py``, and the serving package's
``serving/entry.py``, ``serving/package.py`` and ``serving/__main__.py`` (docs/adr/0012), the
only code that imports mlflow; no method's prediction path imports it.

- ``environment``: ``mlflow_env()``, the only code that sets MLflow's environment, and
  ``use_store()``, which points MLflow at a sqlite store and an experiment. Importing this
  package sets the variables MLflow reads when it is imported (``MLFLOW_IMPORT_ENV``: no
  telemetry, no hint to coding agents), before any of its modules imports mlflow.
- ``pyfunc_model``: the models-from-code entry file every tagging method is logged through.
- ``methods``: the logged model's signature and requirements, ``log_method`` and the run
  lookup, ``standard_metrics``, ``part_metrics`` (each part's, by MLflow dataset) and
  ``log_run_extras``, and ``load_logged_method``, the only sanctioned way to load a logged
  method.
- ``latency``: ``run_latency`` times a frozen method once and logs its summary; a method that
  calls an API is timed on live requests, under a cost cap, except a calibrated twin, which logs
  the ``latency.csv`` copied from its base and times nothing.
- ``evaluation``: the per-problem view, ``ensure_datasets`` (the evaluation datasets, made once)
  and ``evaluate_method`` (one evaluation run per method and part, one trace per problem).
- ``prompts``: the prompt registry: ``register_prompt_versions`` registers a prompt's versions
  once, in a fixed order, then only checks them; ``register_prompts``, its adapter for the
  prompted methods; ``set_prompt_aliases``; ``link_prompt_version`` links a version to a logged
  model and its run.
- ``calibration``: ``run_calibration`` logs the calibration of ``tfidf``'s out-of-fold scores to
  its method run (the table, a metric per estimate and two reliability diagrams), keeping any
  key or file the run already holds; ``calibration_metrics`` names the metrics.
- ``__main__``: ``python -m codeforces_tags.tracking latency``, ``datasets``, ``evaluate``,
  ``discard`` and ``calibrate``.
"""

from codeforces_tags.config import MLFLOW_IMPORT_ENV
from codeforces_tags.tracking.environment import mlflow_env

mlflow_env(MLFLOW_IMPORT_ENV)
