# Figures from MLflow's documentation

Notebook 12 (`notebooks/12_model_serving.ipynb`) shows three figures of the documentation of
MLflow 3.16.1, the version `uv.lock` pins. They were copied here once, unchanged, so that the
notebook needs no network.

| File | Shown on | Copied from | Bytes |
|---|---|---|---|
| `mlflow-deployment-overview.png` | [Deployment](https://mlflow.org/docs/3.16.1/ml/deployment/) | https://mlflow.org/docs/3.16.1/assets/images/mlflow-deployment-overview-f0b56bbb6d5689f022a5cff47f16f832.png | 1,077,158 |
| `models_from_code.png` | [Models](https://mlflow.org/docs/3.16.1/ml/model/) | https://mlflow.org/docs/3.16.1/assets/images/models_from_code-3299c0ebefae1e6e936610a593c27c31.png | 457,951 |
| `signature-enforcement.png` | [Model signatures](https://mlflow.org/docs/3.16.1/ml/model/signatures/) | https://mlflow.org/docs/3.16.1/assets/images/signature-enforcement-e5974f544a264396c51cb2511dd158f1.png | 267,354 |

## Licence

MLflow's documentation is part of the MLflow repository, which is distributed under the Apache
License, Version 2.0, copyright Databricks, Inc. `LICENSE-mlflow.txt` is that licence, as the
`mlflow` 3.16.1 wheel ships it (`mlflow-3.16.1.dist-info/licenses/LICENSE.txt`). The figures are
reproduced unmodified, with this attribution.

## Trademarks

The logos inside the figures (cloud platforms, libraries, MLflow's own) are trademarks of their
owners. They appear here only as MLflow's documentation shows them, and imply no endorsement of
this project.

## One out-of-date label

The deployment overview draws MLflow's model server as a Flask application. In MLflow 3.16.1 the
scoring server is a FastAPI application served by uvicorn (`mlflow.pyfunc.scoring_server`), behind
nginx inside a container; notebook 12 says so where it shows the figure.
