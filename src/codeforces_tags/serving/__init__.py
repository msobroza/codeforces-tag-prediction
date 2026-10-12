"""Serve a frozen tagging method over HTTP with MLflow's model server.

A method is served from its committed method folder (``assets/methods/<name>/model``), never
from the MLflow store: the model server answers each problem as the command line predicts it,
one served record per problem (``records``). CLAUDE.md's layout names every module of the
package.

Importing this package sets the variables MLflow reads when it is imported
(``MLFLOW_IMPORT_ENV``: no telemetry, no hint to coding agents), before any of its modules
imports mlflow.
"""

from codeforces_tags.config import MLFLOW_IMPORT_ENV
from codeforces_tags.tracking.environment import mlflow_env

mlflow_env(MLFLOW_IMPORT_ENV)
