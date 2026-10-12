FROM python:3.12.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt
ARG METHOD
LABEL codeforces-tags.method=${METHOD}
COPY weights /opt/ml/weights
COPY model /opt/ml/model
ENV MLFLOW_MODELS_WORKERS=1 \
    MLFLOW_DISABLE_ENV_CREATION=True \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_ALLOW_PICKLE_DESERIALIZATION=false \
    MLFLOW_UV_AUTO_DETECT=false \
    MLFLOW_LOG_UV_FILES=false \
    MLFLOW_GENAI_EVAL_MAX_WORKERS=1 \
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false \
    MLFLOW_MODEL_CATALOG_URI= \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CODEFORCES_PRETRAINED_DIR=/opt/ml/weights/pretrained \
    CODEFORCES_TRAINED_DIR=/opt/ml/weights/trained \
    CODEFORCES_DEVICE=auto
EXPOSE 8080
ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve('local')"]
