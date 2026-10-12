"""Tune the tag model with Optuna, track the study in MLflow, and freeze the results.

The tuning command has three steps, then two that freeze and verify the tag model it tuned:

    python -m codeforces_tags.tuning run
    python -m codeforces_tags.tuning worker --trials N --seed S
    python -m codeforces_tags.tuning export
    python -m codeforces_tags.tuning freeze
    python -m codeforces_tags.tuning verify

- ``run`` starts ``TUNING_WORKERS`` worker processes and waits for them, then logs every
  complete trial to MLflow: the store ``experiments/mlflow.db`` and the artifacts in
  ``experiments/mlartifacts/``.
- ``worker``, one of these processes, adds trials to the study in the Optuna journal
  ``experiments/optuna_journal.jsonl`` until the study has N complete trials.
- ``export`` writes the frozen results of the best trial to ``assets/tfidf_baseline/``
  (``params.json``, ``thresholds.json``, ``trials.csv``, ``importances.csv`` and
  ``out_of_fold.csv``), then logs the study to MLflow. It fits no tag model.
- ``freeze`` fits the tag model with the frozen configuration, checks that it reproduces the
  frozen results and notebook 04's test score (``TFIDF_TEST_MACRO_F1``), then freezes it once as
  the tagging method ``tfidf`` in ``assets/methods/tfidf/`` and logs it to MLflow.
- ``verify`` checks the frozen files of ``tfidf`` offline, from the files and by fitting again.

``experiments/`` is gitignored; ``assets/tfidf_baseline/`` and ``assets/methods/tfidf/`` are
committed. ``search`` holds the search space and the study (it imports optuna), ``tracking`` the
MLflow logging of the study (it imports mlflow), ``results`` the frozen results, which notebook
04 reads without either library, and ``freeze`` the ``freeze`` and ``verify`` commands (mlflow
only through ``tracking.methods``).
Importing this package sets the variables MLflow reads when it is imported
(``MLFLOW_IMPORT_ENV``: no telemetry, no hint to coding agents), before ``tracking`` imports it.
"""

from codeforces_tags.config import MLFLOW_IMPORT_ENV
from codeforces_tags.tracking.environment import mlflow_env

mlflow_env(MLFLOW_IMPORT_ENV)
