"""The models-from-code entry file every tagging method is logged through (docs/adr/0002).

``log_method`` logs this file by its path, never a ``TaggingMethodModel`` instance: MLflow would
cloudpickle an instance into ``python_model.pkl``. MLflow copies the file into the logged model
and runs the copy when it loads the model: ``set_model`` at the end hands MLflow the wrapper,
whose ``load_context`` loads the method folder logged as the artifact ``method``, through the
registry of the code that loads it (``METHODS``, read when the model loads), never a module
named by the model. Load a logged method with ``tracking.methods.load_logged_method`` only.

Frozen import path: each logged model runs this file's copy, which reaches the framework through
``codeforces_tags.methods.load_method`` and ``codeforces_tags.methods.METHODS`` only. Renaming
or moving them, or this module and its ``TaggingMethodModel``, makes logged models unloadable.
"""

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from codeforces_tags import methods


# predict_stream stays MLflow's: a tagging method answers a batch at once, it does not stream.
class TaggingMethodModel(mlflow.pyfunc.PythonModel):  # pylint: disable=abstract-method
    """MLflow's view of a tagging method: raw problems in, 0/1 tags out."""

    def __init__(self) -> None:
        """Start without a method: MLflow calls ``load_context`` before ``predict``."""
        self.method = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the method folder logged as the artifact ``method``.

        Args:
            context: MLflow's context; ``context.artifacts["method"]`` is the folder's path.
        """
        self.method = methods.load_method(
            Path(context.artifacts["method"]), registry=methods.METHODS
        )

    def predict(  # pylint: disable=unused-argument
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Predict the tags of raw problems.

        Args:
            context: MLflow's context, unused.
            model_input: Raw problems indexed by ``sample_id``; MLflow keeps the columns of the
                signature, the method's ``input_columns``, as strings.
            params: Unused: a tagging method takes no inference parameter.

        Returns:
            The method's int 0/1 tags, with the index of ``model_input``, one column per tag.
        """
        return self.method.predict(model_input)


mlflow.models.set_model(TaggingMethodModel())
