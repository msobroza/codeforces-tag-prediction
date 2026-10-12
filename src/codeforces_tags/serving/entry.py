"""The deployment entry file of a served tagging method: MLflow's model server runs a copy of it.

Packaging a method (``serving package``) saves its committed folder as an MLflow model from code
with this file as its ``python_model``, by its path, never a ``ServedMethodModel`` instance:
MLflow would cloudpickle an instance into ``python_model.pkl``. MLflow copies the file into the
model and runs the copy when the model server loads it, as it runs ``tracking/pyfunc_model.py``
for a logged method. ``set_model`` at the end hands MLflow the wrapper, whose ``load_context``
loads the method folder saved as the artifact ``method`` through the registry of the code that
loads it (``METHODS``, read when the model loads). The served model carries that code, so no
import path of this module is frozen.

Its ``predict`` answers the served records of the problems (``records.served_records``). MLflow's
scoring server runs each request in a thread of its own: the wrapper's lock lets one prediction
run at a time, and so one forward pass on the GPU. A ``CliError`` reaches the client as one
``MlflowException`` with the error's message, which the server answers with status 500, after a
quote of the input, and without a traceback: the error's cause could quote a provider's text.
"""

import threading
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from codeforces_tags import methods
from codeforces_tags.cli_methods import CliError
from codeforces_tags.serving.records import served_records


# predict_stream stays MLflow's: a tagging method answers a batch at once, it does not stream.
class ServedMethodModel(mlflow.pyfunc.PythonModel):  # pylint: disable=abstract-method
    """MLflow's view of a served tagging method: raw problems in, served records out.

    Attributes:
        method: The loaded tagging method; None until ``load_context``.
        spec: Its spec in ``METHODS``; None until ``load_context``.
        lock: Held while a prediction runs: one at a time, whatever the number of requests.
    """

    def __init__(self) -> None:
        """Start without a method: MLflow calls ``load_context`` before ``predict``."""
        self.method = None
        self.spec = None
        self.lock = threading.Lock()

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the method folder saved as the artifact ``method``, and keep the method's spec.

        Args:
            context: MLflow's context; ``context.artifacts["method"]`` is the folder's path.
        """
        registry = methods.METHODS
        self.method = methods.load_method(Path(context.artifacts["method"]), registry=registry)
        self.spec = registry[self.method.name]

    def predict(  # pylint: disable=unused-argument
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Answer the served records of raw problems, one prediction at a time.

        Args:
            context: MLflow's context, unused.
            model_input: Raw problems, as MLflow checked them against the signature: every
                column of the method's ``input_columns`` is ``required`` there, so a request that
                lacks one is refused by MLflow with a 400 and never reaches this method; a null
                value of a column it holds arrives as null, and a column outside the signature is
                dropped.
            params: Unused: a tagging method takes no inference parameter.

        Returns:
            One served record per problem, with the index of ``model_input``
            (``served_records``).

        Raises:
            MlflowException: With the message of the ``CliError`` that ``served_records``
                raised (e.g. two problems sent to a method that calls an API), without its
                cause.
        """
        with self.lock:
            try:
                return served_records(self.method, self.spec, model_input)
            except CliError as error:
                raise MlflowException(str(error)) from None


mlflow.models.set_model(ServedMethodModel())
