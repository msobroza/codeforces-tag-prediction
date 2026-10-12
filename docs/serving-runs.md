# Serving runs

The recorded serving runs that [notebook 12](../notebooks/12_model_serving.ipynb) quotes, taken on the author's machine on 2026-10-11 and 2026-10-12 (UTC). Each record names its command, the date it ran (UTC), its exit status and its duration, then its output. The records fall in three parts: the local servers, `mlflow-ui` and the Streamlit app; Docker, compose and SageMaker's container contract, whose last records are the GPU run of 2026-10-12; and Kubernetes on kind. `<repo>` stands for the repository folder the commands ran in. The commit hashes in the records (among them `1029973` and `324fcda`, the tags of the images) name commits of the private development history, which the public repository squashes into one commit.

Host details that say nothing about the project are left out (the disk's use, uptime and the machine's load, a process id), and each cut is marked in its record. The few log lines stamped in local time are converted to UTC, and numbers are written with decimal points. The records ran through small helper scripts, described at the end (The helper scripts), which are not part of the repository.

## The local servers, `mlflow-ui` and the Streamlit app

### A method whose weights are not on this machine is refused: modernbert_lora

Run on 2026-10-11T21:20:58Z (UTC), exit status 1, in 3 s.

```console
$ uv run --offline python -m codeforces_tags.serving package --method modernbert_lora
python -m codeforces_tags.serving: error: modernbert_lora cannot be packaged on this machine: it needs the pretrained weights of answerdotai/ModernBERT-large in <repo>/experiments/pretrained/answerdotai--ModernBERT-large/45bb4654a4d5aaff24dd11d4781fa46d39bf8c13 (fetch them: `uv run --group finetuning python -m codeforces_tags.methods.modernbert fetch --method modernbert_lora`) and the trained weights in <repo>/experiments/trained/modernbert_lora (copy modernbert_lora/ from the trained cache of the machine that froze the method, docs/adr/0006).
```

### Package tfidf

Run on 2026-10-11T21:21:08Z (UTC), exit status 0, in 6 s.

```console
$ uv run --offline python -m codeforces_tags.serving package --method tfidf
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/3 [00:00<?, ?it/s]
Downloading artifacts:  33%|███▎      | 1/3 [00:00<00:00, 16131.94it/s]
Downloading artifacts:  67%|██████▋   | 2/3 [00:00<00:00, 10305.42it/s]
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1286.07it/s] 
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1186.40it/s]
2026/10/11 21:21:13 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged tfidf in <repo>/build/serving/tfidf
```

### The build folder of tfidf

Run on 2026-10-11T21:21:14Z (UTC), exit status 0, in 0 s.

```console
$ find build/serving/tfidf -maxdepth 2 | sort && du -sh build/serving/tfidf
build/serving/tfidf
build/serving/tfidf/Dockerfile
build/serving/tfidf/model
build/serving/tfidf/model/artifacts
build/serving/tfidf/model/code
build/serving/tfidf/model/conda.yaml
build/serving/tfidf/model/entry.py
build/serving/tfidf/model/MLmodel
build/serving/tfidf/model/python_env.yaml
build/serving/tfidf/model/requirements.txt
build/serving/tfidf/requirements.lock.txt
build/serving/tfidf/weights
6.2M	build/serving/tfidf
```

### Smoke: tfidf on a local server (CPU)

Run on 2026-10-11T21:21:49Z (UTC), exit status 0, in 7 s.

```console
$ uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method tfidf
problem 13: 0.068 s, math: agrees
problem 14: 0.025 s, geometry: agrees
problem 15: 0.019 s, no tag: agrees
problem 18: 0.027 s, no tag: agrees
problem 29: 0.026 s, graphs: agrees
problem 37: 0.024 s, no tag: agrees
problem 39: 0.038 s, no tag: agrees
problem 60: 0.040 s, no tag: agrees
problem 62: 0.040 s, no tag: agrees
problem 64: 0.031 s, graphs: agrees
problem 68: 0.046 s, math: agrees
problem 69: 0.035 s, no tag: agrees
problem 71: 0.026 s, no tag: agrees
problem 74: 0.036 s, math: agrees
problem 75: 0.031 s, no tag: agrees
problem 79: 0.043 s, graphs: agrees
problem 89: 0.044 s, math: agrees
problem 95: 0.031 s, math, probabilities: agrees
problem 99: 0.036 s, no tag: agrees
problem 107: 0.019 s, strings, games: agrees
20 test problems, one request each: median 0.033 s, slowest 0.068 s; 0 differ from their frozen test rows
```

### Package f2llm_late_fusion

Run on 2026-10-11T21:22:13Z (UTC), exit status 0, in 9 s.

```console
$ uv run --offline --group f2llm python -m codeforces_tags.serving package --method f2llm_late_fusion
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/5 [00:00<?, ?it/s]
Downloading artifacts:  20%|██        | 1/5 [00:00<00:00, 13662.23it/s]
Downloading artifacts:  40%|████      | 2/5 [00:00<00:00, 7300.79it/s] 
Downloading artifacts:  60%|██████    | 3/5 [00:00<00:00, 3557.51it/s]
Downloading artifacts:  80%|████████  | 4/5 [00:00<00:00, 3927.25it/s]
Downloading artifacts: 100%|██████████| 5/5 [00:00<00:00, 3544.88it/s]
Downloading artifacts: 100%|██████████| 5/5 [00:00<00:00, 3142.74it/s]
2026/10/11 21:22:18 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged f2llm_late_fusion in <repo>/build/serving/f2llm_late_fusion
```

### The build folder of f2llm_late_fusion

Run on 2026-10-11T21:22:22Z (UTC), exit status 0, in 1 s.

```console
$ find build/serving/f2llm_late_fusion -maxdepth 6 -not -path '*/model/code/*' -not -path '*/model/artifacts/*' | sort && du -sh build/serving/f2llm_late_fusion/* && find build/serving/f2llm_late_fusion/weights -type f | wc -l
build/serving/f2llm_late_fusion
build/serving/f2llm_late_fusion/Dockerfile
build/serving/f2llm_late_fusion/model
build/serving/f2llm_late_fusion/model/artifacts
build/serving/f2llm_late_fusion/model/code
build/serving/f2llm_late_fusion/model/conda.yaml
build/serving/f2llm_late_fusion/model/entry.py
build/serving/f2llm_late_fusion/model/MLmodel
build/serving/f2llm_late_fusion/model/python_env.yaml
build/serving/f2llm_late_fusion/model/requirements.txt
build/serving/f2llm_late_fusion/requirements.lock.txt
build/serving/f2llm_late_fusion/weights
build/serving/f2llm_late_fusion/weights/pretrained
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/1_Pooling
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/1_Pooling/config.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/added_tokens.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/config.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/config_sentence_transformers.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/merges.txt
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/model.safetensors
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/modules.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/special_tokens_map.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/tokenizer_config.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/tokenizer.json
build/serving/f2llm_late_fusion/weights/pretrained/codefuse-ai--F2LLM-0.6B/eb2778d335980b42b986c75d7fe3a78be19ef3a6/vocab.json
4.0K	build/serving/f2llm_late_fusion/Dockerfile
6.1M	build/serving/f2llm_late_fusion/model
228K	build/serving/f2llm_late_fusion/requirements.lock.txt
1.2G	build/serving/f2llm_late_fusion/weights
11
```

### Smoke: f2llm_late_fusion on a local server (CPU)

Run on 2026-10-11T21:32:42Z (UTC), exit status 0, in 115 s.

```console
$ uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method f2llm_late_fusion
problem 13: 3.759 s, no tag: agrees
problem 14: 3.870 s, geometry: agrees
problem 15: 3.357 s, no tag: agrees
problem 18: 6.308 s, no tag: agrees
problem 29: 5.045 s, graphs: agrees
problem 37: 4.775 s, math: agrees
problem 39: 3.694 s, no tag: agrees
problem 60: 9.198 s, strings: agrees
problem 62: 6.957 s, no tag: agrees
problem 64: 4.558 s, graphs: agrees
problem 68: 4.289 s, math: agrees
problem 69: 4.988 s, no tag: agrees
problem 71: 6.108 s, no tag: agrees
problem 74: 11.211 s, no tag: agrees
problem 75: 2.947 s, no tag: agrees
problem 79: 8.593 s, graphs: agrees
problem 89: 6.085 s, math: agrees
problem 95: 7.646 s, math, probabilities: agrees
problem 99: 3.244 s, no tag: agrees
problem 107: 2.302 s, strings, games: agrees
20 test problems, one request each: median 4.881 s, slowest 11.211 s; 0 differ from their frozen test rows
```

### Mirror descode2tag-json and its prompt into the worktree's MLflow store (no request)

Run on 2026-10-11T21:25:34Z (UTC), exit status 0, in 21 s.

```console
$ uv run --offline --group llm python -m codeforces_tags.llm score --method descode2tag-json
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(
4,982 predictions match the frozen files in <repo>/assets/methods/descode2tag-json
per_problem.csv matches the answers files
the method folder <repo>/assets/methods/descode2tag-json/model builds the answered requests

Downloading artifacts:   0%|          | 0/2 [00:00<?, ?it/s]
Downloading artifacts:  50%|█████     | 1/2 [00:00<00:00, 2857.16it/s]
Downloading artifacts: 100%|██████████| 2/2 [00:00<00:00, 3778.65it/s]
Downloading artifacts: 100%|██████████| 2/2 [00:00<00:00, 3287.07it/s]
2026/10/11 21:25:54 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
<repo>/.venv/lib/python3.12/site-packages/mlflow/types/utils.py:440: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.
  warnings.warn(
<repo>/.venv/lib/python3.12/site-packages/mlflow/types/utils.py:440: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.
  warnings.warn(
logged descode2tag-json to MLflow: model m-3fe13544737a43d89007aeedb9f67983, prompt prompts:/descode2tag/2
```

### Package descode2tag-json, a method that calls an API (no request)

Run on 2026-10-11T21:26:01Z (UTC), exit status 0, in 6 s.

```console
$ uv run --offline --group llm python -m codeforces_tags.serving package --method descode2tag-json && find build/serving/descode2tag-json -maxdepth 2 | sort && grep -E '^(openai|mlflow)==' build/serving/descode2tag-json/model/requirements.txt
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/2 [00:00<?, ?it/s]
Downloading artifacts:  50%|█████     | 1/2 [00:00<00:00, 859.66it/s]
Downloading artifacts: 100%|██████████| 2/2 [00:00<00:00, 1368.23it/s]
Downloading artifacts: 100%|██████████| 2/2 [00:00<00:00, 1210.83it/s]
2026/10/11 21:26:06 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged descode2tag-json in <repo>/build/serving/descode2tag-json
build/serving/descode2tag-json
build/serving/descode2tag-json/Dockerfile
build/serving/descode2tag-json/model
build/serving/descode2tag-json/model/artifacts
build/serving/descode2tag-json/model/code
build/serving/descode2tag-json/model/conda.yaml
build/serving/descode2tag-json/model/entry.py
build/serving/descode2tag-json/model/MLmodel
build/serving/descode2tag-json/model/python_env.yaml
build/serving/descode2tag-json/model/requirements.txt
build/serving/descode2tag-json/requirements.lock.txt
build/serving/descode2tag-json/weights
openai==3.26.0
mlflow==3.16.1
```

### serving mlflow-ui: the worktree's store on 127.0.0.1:5000

Run on 2026-10-11T21:26:28Z (UTC), exit status 0, in 0 s.

```console
$ head -n 5 experiments/serving-runs/mlflow-ui.log
the tag-methods experiment: http://127.0.0.1:5000/#/experiments/1
the prompt registry: http://127.0.0.1:5000/#/prompts
Registry store URI not provided. Using backend store URI.
[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
<repo>/.venv/lib/python3.12/site-packages/mlflow/server/fastapi_app.py:23: StarletteDeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to https://github.com/abersheeran/a2wsgi as a replacement.
```

### The links of three methods, as the app's sidebar gets them

Run on 2026-10-11T21:26:28Z (UTC), exit status 0, in 3 s.

```console
$ uv run --offline python experiments/serving-runs/links.py
MLflow UI at http://127.0.0.1:5000: up
f2llm_late_fusion
  experiment: http://127.0.0.1:5000/#/experiments/1
  run:        http://127.0.0.1:5000/#/experiments/1/runs/66dbc6ac56e84151838b0d55b45210cf
  registry:   http://127.0.0.1:5000/#/prompts
tfidf
  experiment: http://127.0.0.1:5000/#/experiments/1
  run:        not logged in this store
  registry:   http://127.0.0.1:5000/#/prompts
descode2tag-json
  experiment: http://127.0.0.1:5000/#/experiments/1
  run:        http://127.0.0.1:5000/#/experiments/1/runs/e8898b8740114d1eb8e09a1d5e222f53
  prompt:     descode2tag v2 http://127.0.0.1:5000/#/prompts/descode2tag?promptVersion=2
  registry:   http://127.0.0.1:5000/#/prompts
```

### The app's script, run once without a browser (Streamlit's AppTest)

Run on 2026-10-11T21:28:15Z (UTC), exit status 0, in 7 s.

```console
$ uv run --offline --group app python experiments/serving-runs/app_check.py
2026-10-11 21:28:16.183 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
sidebar text_input: http://127.0.0.1:8080
sidebar text_input: http://127.0.0.1:5000
sidebar caption: Served tagging method: known once the model server answers
sidebar caption: Predict once: the links follow the tagging method the server serves.
sidebar success: It answers /ping.
sidebar success: It answers /health.
buttons: ['Predict']
```

### The Streamlit app and the MLflow links, checked by hand

Checked on 2026-10-11 from 21:26 to 21:37 (UTC), in a browser (Chromium, a 1400x900 viewport), against the late-fusion local server on the CPU (`serve_local.py --method f2llm_late_fusion --port 8080`), `serving mlflow-ui` on port 5000 and `streamlit run apps/streamlit_app.py` on port 8501.

- The experiment link, `http://127.0.0.1:5000/#/experiments/1`: ok. It opens the experiment `tag-methods` (MLflow 3.16.1 opens an experiment on its Overview page); its Training runs list holds 10 runs: `descode2tag-json`, `f2llm_probe`, `f2llm_late_fusion`, `f2llm_early_fusion` and the six `eval-f2llm_*-dev` and `eval-f2llm_*-test` runs.
- The run of `f2llm_late_fusion`, `http://127.0.0.1:5000/#/experiments/1/runs/66dbc6ac56e84151838b0d55b45210cf`: ok, with one difference in where things are shown. The page is the run `f2llm_late_fusion`, with its 30 metrics (the dev out-of-fold scores, macro F1 0.7057, per-tag F1, precision and recall, and the frozen latency metrics: median 3.03 s, slowest 8.48 s, loading 7.81 s) and "Parameters (0)". The run record's entries (75 parameters) and the test metrics (macro F1 0.7477), beside the dev out-of-fold ones, are on the run's logged model `tagging_method`, one click from the run: `log_method` logs them there.
- The run of `descode2tag-json`, `http://127.0.0.1:5000/#/experiments/1/runs/e8898b8740114d1eb8e09a1d5e222f53`: ok, the same layout. It is the run `descode2tag-json`, logged by the record "Mirror descode2tag-json and its prompt into the worktree's MLflow store" above, from commit `117f154`, with 29 metrics (dev out of fold, macro F1 0.7319) and "Registered prompts: descode2tag (v2)". As for late fusion, "Parameters (0)" is on the run, and the run record is on its logged model.
- The prompt link, `http://127.0.0.1:5000/#/prompts/descode2tag?promptVersion=2`: ok. It is the prompt `descode2tag`, with versions 1 and 2 and version 2 viewed (alias `json`, commit message "descode2tag-json: the json output format", metadata `content_sha256`), its model config, its structured-output schema and its template.
- The registry, `http://127.0.0.1:5000/#/prompts`: ok. The list of prompts holds `descode2tag`, latest version 2.
- (a) ok. The sidebar shows the server `http://127.0.0.1:8080` with "It answers /ping." and, under "MLflow tracking", the UI `http://127.0.0.1:5000` with "It answers /health.".
- (b) ok, with one display difference. Mode "Test-set problem" lists 994 test problems ("Problem (994 of the 994 test-set problems)"), and 33 once the true focus tag `geometry` is chosen. Problem 170 shows its statement (Description, Input specification, Output specification), its notes, its two sample input and output pairs and its solution. Its `$$$...$$$` formulas are typeset as `$...$` math. The difference: the dataset writes `<` as the HTML entity `&lt;` inside some formulas (problem 170: `$$$h_1 &lt; h_2$$$`), which the typesetter cannot read, so it shows `h_1 &lt; h_2` in red.
- (c) ok. Predict on problem 170 shows "Prediction of f2llm_late_fusion". Predicted focus tags: geometry; true focus tags: geometry; frozen prediction: geometry. The table has the columns tag, predicted, score, threshold, true, frozen and agrees, and all 8 tags agree. The Altair chart has 8 score bars and 8 black ticks: every per-tag threshold of `f2llm_late_fusion` is finite, so every tag has its tick. Its axis title is "score", and the page never says "probability" (only the tag `probabilities`). The round trip was 7.63 s: the first request after the server started, on a machine loaded by another job (see the record "Smoke: f2llm_late_fusion on a local server (CPU)" above). The caption reads "A demonstration on frozen test-set problems: it computes no metric and chooses nothing (ADR 0001); every tagging method was chosen on the dev set."
- (d) ok. After that answer, the sidebar reads "Served tagging method: f2llm_late_fusion" and shows three link buttons: "Experiment tag-methods" (`http://127.0.0.1:5000/#/experiments/1`), "Run of f2llm_late_fusion" (`.../#/experiments/1/runs/66dbc6ac56e84151838b0d55b45210cf`) and "Prompt registry" (`http://127.0.0.1:5000/#/prompts`). There is no prompt version. Each button is the page checked above, and "Run of f2llm_late_fusion", clicked, opened that run's page.
- (e) ok. Mode "New problem" shows five text areas (Description (required), Input specification, Output specification, Notes, Solution (required)) and a data editor of (Sample input, Sample output) rows. The form got a description and an input specification with `$$$` math and a Python solution, no notes, one pair whose sample output is blank (`3`, blank) and one blank row. Predict shows "Predicted focus tags: geometry" and a round trip of 1.97 s, with a table of tag, predicted, score and threshold only: no truth and no frozen prediction. With the description blanked, Predict shows one message, "No prediction: the description is empty: a problem needs a description and a solution.", and no traceback.
- (f) ok. `stop.sh serve-late-fusion` printed "stopped serve-late-fusion (process group ...)", then "port 8080 is free". Predict then shows one error, "No prediction: no answer from the model server at http://127.0.0.1:8080/invocations: [Errno 111] Connection refused", and no traceback. The sidebar's status reads "It does not answer /ping: no model server runs there, or it is still loading its tagging method.", followed by the commands that build and start a server. Until the reload of (g), it still named the last served method, `f2llm_late_fusion`, with its link buttons.
- (g) ok. `stop.sh mlflow-ui` printed "stopped mlflow-ui (process group ...)", and port 5000 was free. After a reload, the sidebar says "The MLflow UI does not answer at http://127.0.0.1:5000. Start it with:" and shows `uv run python -m codeforces_tags.serving mlflow-ui`. There are no link buttons and no exception.

## Docker, compose and SageMaker's container contract

### What each build downloads from PyPI (sizes from uv.lock)

Run on 2026-10-11T22:25:04Z (UTC), exit status 0, in 0 s.

```console
$ uv run --offline python experiments/serving-runs/download_sizes.py --requirements build/serving/tfidf/requirements.lock.txt --top 3 && uv run --offline python experiments/serving-runs/download_sizes.py --requirements build/serving/f2llm_late_fusion/requirements.lock.txt --top 5
pyarrow==25.0.1: 50.1 MB
scipy==1.18.1: 35.3 MB
numpy==2.5.3: 16.7 MB
88 packages, 187.7 MB to download (sizes from uv.lock)
torch==2.14.1: 554.6 MB
nvidia-cudnn-cu13==9.24.0.43: 553.1 MB
nvidia-cublas==13.1.1.3: 423.1 MB
triton==3.8.0: 248.0 MB
nvidia-nccl-cu13==2.30.7: 216.0 MB
132 packages, 3223.8 MB to download (sizes from uv.lock)
```

### Build the tfidf image

Run on 2026-10-11T22:25:10Z (UTC), exit status 0, in 61 s.

```console
$ uv run --offline python -m codeforces_tags.serving build --method tfidf
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/3 [00:00<?, ?it/s]
Downloading artifacts:  33%|███▎      | 1/3 [00:00<00:00, 4378.19it/s]
Downloading artifacts:  67%|██████▋   | 2/3 [00:00<00:00, 4277.72it/s]
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1229.88it/s]
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1193.82it/s]
2026/10/11 22:25:13 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged tfidf in <repo>/build/serving/tfidf
python -m codeforces_tags.serving: note: building codeforces-tags/tfidf:1029973: its log is printed only if the build fails
built codeforces-tags/tfidf:1029973 and codeforces-tags/tfidf:latest: image ce22ca2389c9, 0.91 GB
kept codeforces-tags/tfidf:1029973, codeforces-tags/tfidf:latest (image ce22ca2389c9)
pruned the dangling images labelled codeforces-tags.method=tfidf: Total reclaimed space: 0B
dropped BuildKit's cached build contexts: Total:	5.908MB
deleted the staged weights, which the image holds: <repo>/build/serving/tfidf/weights
```

### The tfidf image and its layers

Run on 2026-10-11T22:26:16Z (UTC), exit status 0, in 0 s.

```console
$ docker image ls codeforces-tags/tfidf --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' && docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' codeforces-tags/tfidf:latest | cut -c1-150
REPOSITORY:TAG                  IMAGE ID       SIZE
codeforces-tags/tfidf:1029973   ce22ca2389c9   906MB
codeforces-tags/tfidf:latest    ce22ca2389c9   906MB
0B	ENTRYPOINT ["python" "-c" "from mlflow.models import container as C; C._serve('local')"]
0B	EXPOSE map[8080/tcp:{}]
0B	ENV MLFLOW_MODELS_WORKERS=1 MLFLOW_DISABLE_ENV_CREATION=True MLFLOW_DISABLE_TELEMETRY=true MLFLOW_ALLOW_PICKLE_DESERIALIZATION=false MLFLOW_UV_AUTO
5.71MB	COPY model /opt/ml/model # buildkit
0B	COPY weights /opt/ml/weights # buildkit
0B	LABEL codeforces-tags.method=tfidf
0B	ARG METHOD=tfidf
770MB	RUN /bin/sh -c pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt     && rm /tmp/requirements.lock.txt # buildkit
200kB	COPY requirements.lock.txt /tmp/requirements.lock.txt # buildkit
10.3MB	RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends nginx     && rm -rf /var/lib/apt/lists/* # buildkit
0B	ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
0B	CMD ["python3"]
36B	RUN /bin/sh -c set -eux;  for src in idle3 pip3 pydoc3 python3 python3-config; do   dst="$(echo "$src" | tr -d 3)";   [ -s "/usr/local/bin/$src" ]
36.8MB	RUN /bin/sh -c set -eux;   savedAptMark="$(apt-mark showmanual)";  apt-get update;  apt-get install -y --no-install-recommends   dpkg-dev   gcc
0B	ENV PYTHON_SHA256=fb85a13414b028c49ba18bbd523c2d055a30b56b18b92ce454ea2c51edc656c4
0B	ENV PYTHON_VERSION=3.12.12
0B	ENV GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305
3.81MB	RUN /bin/sh -c set -eux;  apt-get update;  apt-get install -y --no-install-recommends   ca-certificates   netbase   tzdata  ;  apt-get dist-cle
0B	ENV LANG=C.UTF-8
0B	ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
78.6MB	# debian.sh --arch 'amd64' out/ 'trixie' '@1771804800'
```

### Build tfidf again: build removes the image it replaces

Run on 2026-10-11T22:26:22Z (UTC), exit status 0, in 5 s.

```console
$ uv run --offline python -m codeforces_tags.serving build --method tfidf
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/3 [00:00<?, ?it/s]
Downloading artifacts:  33%|███▎      | 1/3 [00:00<00:00, 19152.07it/s]
Downloading artifacts:  67%|██████▋   | 2/3 [00:00<00:00, 3184.74it/s] 
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1588.55it/s]
Downloading artifacts: 100%|██████████| 3/3 [00:00<00:00, 1489.45it/s]
2026/10/11 22:26:25 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged tfidf in <repo>/build/serving/tfidf
python -m codeforces_tags.serving: note: building codeforces-tags/tfidf:1029973: its log is printed only if the build fails
built codeforces-tags/tfidf:1029973 and codeforces-tags/tfidf:latest: image 8af058113b18, 0.91 GB
kept codeforces-tags/tfidf:1029973, codeforces-tags/tfidf:latest (image 8af058113b18)
pruned the dangling images labelled codeforces-tags.method=tfidf: Total reclaimed space: 0B
dropped BuildKit's cached build contexts: Total:	5.908MB
deleted the staged weights, which the image holds: <repo>/build/serving/tfidf/weights
```

### The images labelled codeforces-tags.method=tfidf after the second build

Run on 2026-10-11T22:26:27Z (UTC), exit status 0, in 0 s.

```console
$ docker image ls --filter label=codeforces-tags.method=tfidf --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}\t{{.Size}}'
REPOSITORY:TAG                  IMAGE ID       CREATED        SIZE
codeforces-tags/tfidf:1029973   8af058113b18   1 second ago   906MB
codeforces-tags/tfidf:latest    8af058113b18   1 second ago   906MB
```

### Build the f2llm_late_fusion image

Run on 2026-10-11T22:26:41Z (UTC), exit status 0, in 188 s.

```console
$ uv run --offline --group f2llm python -m codeforces_tags.serving build --method f2llm_late_fusion
<repo>/.venv/lib/python3.12/site-packages/mlflow/pyfunc/__init__.py:3344: UserWarning: An input example was not provided when logging the model. To ensure the model signature functions correctly, specify the `input_example` parameter. See https://mlflow.org/docs/latest/model/signatures.html#model-input-example for more details about the benefits of using input_example.
  color_warning(

Downloading artifacts:   0%|          | 0/5 [00:00<?, ?it/s]
Downloading artifacts:  20%|██        | 1/5 [00:00<00:00, 15827.56it/s]
Downloading artifacts:  40%|████      | 2/5 [00:00<00:00, 4359.98it/s] 
Downloading artifacts:  60%|██████    | 3/5 [00:00<00:00, 3656.76it/s]
Downloading artifacts:  80%|████████  | 4/5 [00:00<00:00, 4161.02it/s]
Downloading artifacts: 100%|██████████| 5/5 [00:00<00:00, 1448.51it/s]
Downloading artifacts: 100%|██████████| 5/5 [00:00<00:00, 1398.19it/s]
2026/10/11 22:26:44 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
packaged f2llm_late_fusion in <repo>/build/serving/f2llm_late_fusion
python -m codeforces_tags.serving: note: building codeforces-tags/f2llm_late_fusion:1029973: its log is printed only if the build fails
built codeforces-tags/f2llm_late_fusion:1029973 and codeforces-tags/f2llm_late_fusion:latest: image 26be1c73ce3a, 7.89 GB
kept codeforces-tags/f2llm_late_fusion:1029973, codeforces-tags/f2llm_late_fusion:latest (image 26be1c73ce3a)
pruned the dangling images labelled codeforces-tags.method=f2llm_late_fusion: Total reclaimed space: 0B
dropped BuildKit's cached build contexts: Total:	1.214GB
deleted the staged weights, which the image holds: <repo>/build/serving/f2llm_late_fusion/weights
```

### The f2llm_late_fusion image and its layers

Run on 2026-10-11T22:30:00Z (UTC), exit status 0, in 0 s.

```console
$ docker image ls codeforces-tags/f2llm_late_fusion --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' && docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' codeforces-tags/f2llm_late_fusion:latest | cut -c1-150 && df -h /
REPOSITORY:TAG                              IMAGE ID       SIZE
codeforces-tags/f2llm_late_fusion:1029973   26be1c73ce3a   7.89GB
codeforces-tags/f2llm_late_fusion:latest    26be1c73ce3a   7.89GB
0B	ENTRYPOINT ["python" "-c" "from mlflow.models import container as C; C._serve('local')"]
0B	EXPOSE map[8080/tcp:{}]
0B	ENV MLFLOW_MODELS_WORKERS=1 MLFLOW_DISABLE_ENV_CREATION=True MLFLOW_DISABLE_TELEMETRY=true MLFLOW_ALLOW_PICKLE_DESERIALIZATION=false MLFLOW_UV_AUTO
5.86MB	COPY model /opt/ml/model # buildkit
1.21GB	COPY weights /opt/ml/weights # buildkit
0B	LABEL codeforces-tags.method=f2llm_late_fusion
0B	ARG METHOD=f2llm_late_fusion
6.55GB	RUN /bin/sh -c pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt     && rm /tmp/requirements.lock.txt # buildkit
230kB	COPY requirements.lock.txt /tmp/requirements.lock.txt # buildkit
10.3MB	RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends nginx     && rm -rf /var/lib/apt/lists/* # buildkit
0B	ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
0B	CMD ["python3"]
36B	RUN /bin/sh -c set -eux;  for src in idle3 pip3 pydoc3 python3 python3-config; do   dst="$(echo "$src" | tr -d 3)";   [ -s "/usr/local/bin/$src" ]
36.8MB	RUN /bin/sh -c set -eux;   savedAptMark="$(apt-mark showmanual)";  apt-get update;  apt-get install -y --no-install-recommends   dpkg-dev   gcc
0B	ENV PYTHON_SHA256=fb85a13414b028c49ba18bbd523c2d055a30b56b18b92ce454ea2c51edc656c4
0B	ENV PYTHON_VERSION=3.12.12
0B	ENV GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305
3.81MB	RUN /bin/sh -c set -eux;  apt-get update;  apt-get install -y --no-install-recommends   ca-certificates   netbase   tzdata  ;  apt-get dist-cle
0B	ENV LANG=C.UTF-8
0B	ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
78.6MB	# debian.sh --arch 'amd64' out/ 'trixie' '@1771804800'
[df's 2 lines left out]
```

### Compose, profile cpu: up, until the healthcheck passes

Run on 2026-10-11T22:30:07Z (UTC), exit status 0, in 17 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile cpu up -d --wait --wait-timeout 300
 Network codeforces-tags_default  Creating
 Network codeforces-tags_default  Created
 Container codeforces-tags-model-server-cpu-1  Creating
 Container codeforces-tags-model-server-cpu-1  Created
 Container codeforces-tags-model-server-cpu-1  Starting
 Container codeforces-tags-model-server-cpu-1  Started
 Container codeforces-tags-model-server-cpu-1  Waiting
 Container codeforces-tags-model-server-cpu-1  Healthy
```

### Compose, profile cpu: the service

Run on 2026-10-11T22:30:24Z (UTC), exit status 0, in 0 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile cpu ps
NAME                                 IMAGE                                      COMMAND                  SERVICE            CREATED          STATUS                    PORTS
codeforces-tags-model-server-cpu-1   codeforces-tags/f2llm_late_fusion:latest   "python -c 'from mlf…"   model-server-cpu   17 seconds ago   Up 16 seconds (healthy)   127.0.0.1:8080->8080/tcp
```

### Smoke: f2llm_late_fusion under compose, profile cpu

Run on 2026-10-11T22:30:30Z (UTC), exit status 0, in 83 s.

```console
$ uptime && uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method f2llm_late_fusion && uptime
[uptime's line left out]
problem 13: 2.402 s, no tag: agrees
problem 14: 2.501 s, geometry: agrees
problem 15: 2.415 s, no tag: agrees
problem 18: 4.643 s, no tag: agrees
problem 29: 3.400 s, graphs: agrees
problem 37: 3.391 s, math: agrees
problem 39: 2.467 s, no tag: agrees
problem 60: 6.705 s, strings: agrees
problem 62: 5.514 s, no tag: agrees
problem 64: 2.975 s, graphs: agrees
problem 68: 2.552 s, math: agrees
problem 69: 3.975 s, no tag: agrees
problem 71: 4.717 s, no tag: agrees
problem 74: 8.207 s, no tag: agrees
problem 75: 1.848 s, no tag: agrees
problem 79: 6.653 s, graphs: agrees
problem 89: 4.785 s, math: agrees
problem 95: 6.262 s, math, probabilities: agrees
problem 99: 2.246 s, no tag: agrees
problem 107: 1.355 s, strings, games: agrees
20 test problems, one request each: median 3.395 s, slowest 8.207 s; 0 differ from their frozen test rows
[uptime's line left out]
```

### Compose, profile cpu: the end of the server's log

Run on 2026-10-11T22:31:59Z (UTC), exit status 0, in 0 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile cpu logs --no-color --tail 25
model-server-cpu-1  | 127.0.0.1 - - [11/Oct/2026:22:31:16 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:21 +0000] "POST /invocations HTTP/1.1" 200 823 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-cpu-1  | 127.0.0.1 - - [11/Oct/2026:22:31:27 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:29 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:31 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-cpu-1  | 127.0.0.1 - - [11/Oct/2026:22:31:38 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:38 +0000] "POST /invocations HTTP/1.1" 200 824 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:43 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-cpu-1  | 127.0.0.1 - - [11/Oct/2026:22:31:48 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:49 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:51 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     172.18.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-cpu-1  | 172.18.0.1 - - [11/Oct/2026:22:31:53 +0000] "POST /invocations HTTP/1.1" 200 819 "-" "Python-urllib/3.12"
model-server-cpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-cpu-1  | 127.0.0.1 - - [11/Oct/2026:22:31:59 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
```

### Compose, profile cpu: down

Run on 2026-10-11T22:31:59Z (UTC), exit status 0, in 1 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile cpu down
 Container codeforces-tags-model-server-cpu-1  Stopping
 Container codeforces-tags-model-server-cpu-1  Stopped
 Container codeforces-tags-model-server-cpu-1  Removing
 Container codeforces-tags-model-server-cpu-1  Removed
 Network codeforces-tags_default  Removing
 Network codeforces-tags_default  Removed
```

### SageMaker's contract, run locally: docker run with the argument serve

Run on 2026-10-11T22:32:07Z (UTC), exit status 0, in 1 s.

```console
$ docker run -d --rm --name codeforces-tags-serve --pull never -p 127.0.0.1:8080:8080 codeforces-tags/f2llm_late_fusion:1029973 serve
93a87a0c3945d8efcb0607869a5334c69a9210a9f1f775311707d2d8be866bc3
```

### /ping, and a path that nginx does not serve

Run on 2026-10-11T22:32:29Z (UTC), exit status 0, in 0 s.

```console
$ curl --noproxy '*' -s -o /dev/null -w '/ping %{http_code}\n' http://127.0.0.1:8080/ping && curl --noproxy '*' -s -w ' /other %{http_code}\n' http://127.0.0.1:8080/other
/ping 200
{} /other 404
```

### Smoke: f2llm_late_fusion under docker run ... serve

Run on 2026-10-11T22:32:29Z (UTC), exit status 0, in 84 s.

```console
$ uptime && uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method f2llm_late_fusion && uptime
[uptime's line left out]
problem 13: 2.248 s, no tag: agrees
problem 14: 2.717 s, geometry: agrees
problem 15: 2.364 s, no tag: agrees
problem 18: 3.822 s, no tag: agrees
problem 29: 3.732 s, graphs: agrees
problem 37: 3.214 s, math: agrees
problem 39: 2.492 s, no tag: agrees
problem 60: 7.571 s, strings: agrees
problem 62: 5.181 s, no tag: agrees
problem 64: 3.467 s, graphs: agrees
problem 68: 3.252 s, math: agrees
problem 69: 4.308 s, no tag: agrees
problem 71: 4.741 s, no tag: agrees
problem 74: 7.995 s, no tag: agrees
problem 75: 2.383 s, no tag: agrees
problem 79: 7.407 s, graphs: agrees
problem 89: 4.734 s, math: agrees
problem 95: 4.752 s, math, probabilities: agrees
problem 99: 2.561 s, no tag: agrees
problem 107: 1.763 s, strings, games: agrees
20 test problems, one request each: median 3.600 s, slowest 7.995 s; 0 differ from their frozen test rows
[uptime's line left out]
```

### The container's log, then docker stop

Run on 2026-10-11T22:33:58Z (UTC), exit status 0, in 1 s.

```console
$ docker logs --tail 15 codeforces-tags-serve 2>&1 && docker stop codeforces-tags-serve
172.17.0.1 - - [11/Oct/2026:22:33:21 +0000] "POST /invocations HTTP/1.1" 200 823 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:29 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:32 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:39 +0000] "POST /invocations HTTP/1.1" 200 824 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:44 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:49 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:51 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
INFO:     172.17.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
172.17.0.1 - - [11/Oct/2026:22:33:53 +0000] "POST /invocations HTTP/1.1" 200 819 "-" "Python-urllib/3.12"
codeforces-tags-serve
```

### The images after the Docker runs on the CPU

Run on 2026-10-11T22:34:09Z (UTC), exit status 0, in 0 s.

```console
$ docker image ls --filter label=codeforces-tags.method --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' && docker system df && df -h /
REPOSITORY:TAG                              IMAGE ID       SIZE
codeforces-tags/f2llm_late_fusion:1029973   26be1c73ce3a   7.89GB
codeforces-tags/f2llm_late_fusion:latest    26be1c73ce3a   7.89GB
codeforces-tags/tfidf:1029973               8af058113b18   906MB
codeforces-tags/tfidf:latest                8af058113b18   906MB
[docker system df's 5 lines and df's 2 lines left out]
```

### Compose, profile gpu: the NVIDIA Container Toolkit, and the GPU seen from the image

Run on 2026-10-12T02:22:01Z (UTC), exit status 0, in 1 s.

```console
$ nvidia-ctk --version | head -1 && docker run --rm --gpus all --pull never --entrypoint nvidia-smi codeforces-tags/f2llm_late_fusion:324fcda --query-gpu=name,driver_version,memory.used,temperature.gpu --format=csv
NVIDIA Container Toolkit CLI version 1.20.1
name, driver_version, memory.used [MiB], temperature.gpu
NVIDIA GeForce RTX 2080 Ti, 580.178.04, 6 MiB, 38
```

### Compose, profile gpu: up, until the healthcheck passes

Run on 2026-10-12T02:22:02Z (UTC), exit status 0, in 16 s.

```console
$ TAG=324fcda docker compose -f deploy/docker-compose.yml --profile gpu up -d --wait --wait-timeout 300
 Network codeforces-tags_default  Creating
 Network codeforces-tags_default  Created
 Container codeforces-tags-model-server-gpu-1  Creating
 Container codeforces-tags-model-server-gpu-1  Created
 Container codeforces-tags-model-server-gpu-1  Starting
 Container codeforces-tags-model-server-gpu-1  Started
 Container codeforces-tags-model-server-gpu-1  Waiting
 Container codeforces-tags-model-server-gpu-1  Healthy
```

### Compose, profile gpu: the service, and the GPU memory its process holds

Run on 2026-10-12T02:22:24Z (UTC), exit status 0, in 1 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile gpu ps && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv && nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv
NAME                                 IMAGE                                       COMMAND                  SERVICE            CREATED          STATUS                    PORTS
codeforces-tags-model-server-gpu-1   codeforces-tags/f2llm_late_fusion:324fcda   "python -c 'from mlf…"   model-server-gpu   23 seconds ago   Up 22 seconds (healthy)   127.0.0.1:8080->8080/tcp
[the pid column left out]
process_name, used_gpu_memory [MiB]
/usr/local/bin/python3.12, 2432 MiB
memory.used [MiB], temperature.gpu, power.draw [W]
2441 MiB, 43, 57.42 W
```

### Smoke: one problem first, under compose, profile gpu

Run on 2026-10-12T02:22:37Z (UTC), exit status 0, in 6 s.

```console
$ uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method f2llm_late_fusion --count 1
problem 13: 0.366 s, no tag: agrees
1 test problems, one request each: median 0.366 s, slowest 0.366 s; 0 differ from their frozen test rows
```

### Smoke: f2llm_late_fusion under compose, profile gpu

Run on 2026-10-12T02:22:54Z (UTC), exit status 0, in 7 s.

```console
$ uptime && uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:8080 --method f2llm_late_fusion && uptime
[uptime's line left out]
problem 13: 0.140 s, no tag: agrees
problem 14: 0.149 s, geometry: agrees
problem 15: 0.130 s, no tag: agrees
problem 18: 0.183 s, no tag: agrees
problem 29: 0.148 s, graphs: agrees
problem 37: 0.147 s, math: agrees
problem 39: 0.122 s, no tag: agrees
problem 60: 0.228 s, strings: agrees
problem 62: 0.185 s, no tag: agrees
problem 64: 0.133 s, graphs: agrees
problem 68: 0.138 s, math: agrees
problem 69: 0.145 s, no tag: agrees
problem 71: 0.162 s, no tag: agrees
problem 74: 0.262 s, no tag: agrees
problem 75: 0.110 s, no tag: agrees
problem 79: 0.211 s, graphs: agrees
problem 89: 0.167 s, math: agrees
problem 95: 0.195 s, math, probabilities: agrees
problem 99: 0.109 s, no tag: agrees
problem 107: 0.102 s, strings, games: agrees
20 test problems, one request each: median 0.148 s, slowest 0.262 s; 0 differ from their frozen test rows
[uptime's line left out]
```

### Compose, profile gpu: the end of the server's log, and the GPU's peaks (a watchdog sampled temperature and power every 250 ms, from before up until down)

Run on 2026-10-12T02:23:24Z (UTC), exit status 0, in 0 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile gpu logs --no-color --tail 25 && awk '{if ($2 + 0 > t) t = $2 + 0; if ($3 + 0 > p) p = $3 + 0; if ($3 + 0 > 230) h++} END {printf "%d samples: peak %d C, peak %.0f W, %d samples above 230 W\n", NR, t, p, h}' experiments/serving-runs/gpu-watchdog.log
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:22:59 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:22:59 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-gpu-1  | 127.0.0.1 - - [12/Oct/2026:02:22:59 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:22:59 +0000] "POST /invocations HTTP/1.1" 200 825 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:00 +0000] "POST /invocations HTTP/1.1" 200 823 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:00 +0000] "POST /invocations HTTP/1.1" 200 822 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:00 +0000] "POST /invocations HTTP/1.1" 200 825 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:00 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:00 +0000] "POST /invocations HTTP/1.1" 200 817 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:01 +0000] "POST /invocations HTTP/1.1" 200 820 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     172.20.0.1:0 - "POST /invocations HTTP/1.0" 200 OK
model-server-gpu-1  | 172.20.0.1 - - [12/Oct/2026:02:23:01 +0000] "POST /invocations HTTP/1.1" 200 821 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-gpu-1  | 127.0.0.1 - - [12/Oct/2026:02:23:09 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
model-server-gpu-1  | INFO:     127.0.0.1:0 - "GET /ping HTTP/1.0" 200 OK
model-server-gpu-1  | 127.0.0.1 - - [12/Oct/2026:02:23:20 +0000] "GET /ping HTTP/1.1" 200 1 "-" "Python-urllib/3.12"
357 samples: peak 56 C, peak 252 W, 3 samples above 230 W
```

### Compose, profile gpu: down

Run on 2026-10-12T02:23:24Z (UTC), exit status 0, in 1 s.

```console
$ docker compose -f deploy/docker-compose.yml --profile gpu down
 Container codeforces-tags-model-server-gpu-1  Stopping
 Container codeforces-tags-model-server-gpu-1  Stopped
 Container codeforces-tags-model-server-gpu-1  Removing
 Container codeforces-tags-model-server-gpu-1  Removed
 Network codeforces-tags_default  Removing
 Network codeforces-tags_default  Removed
```

### The GPU as the image sees it

Run on 2026-10-12T02:24:59Z (UTC), exit status 0, in 3 s.

```console
$ docker run --rm --pull never --gpus all --entrypoint python codeforces-tags/f2llm_late_fusion:latest -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
2.14.1+cu130 13.0 True NVIDIA GeForce RTX 2080 Ti
```

### The watchdog's samples while the smoke of 20 problems ran (UTC time, temperature in C, power in W)

Run on 2026-10-12T03:06:44Z (UTC), exit status 0, in 0 s.

```console
$ awk '$1 >= "02:22:54" && $1 <= "02:23:02"' experiments/serving-runs/gpu-watchdog.log
02:22:54.153 44 58.36
02:22:54.404 44 58.24
02:22:54.656 44 58.27
02:22:54.909 44 58.12
02:22:55.159 45 58.18
02:22:55.449 45 58.21
02:22:55.662 45 58.29
02:22:55.913 45 58.28
02:22:56.205 44 58.14
02:22:56.415 44 58.45
02:22:56.667 45 58.39
02:22:56.919 45 58.14
02:22:57.170 45 58.34
02:22:57.422 45 58.27
02:22:57.673 46 74.24
02:22:57.929 46 104.79
02:22:58.179 50 172.87
02:22:58.434 50 165.38
02:22:58.704 50 205.71
02:22:58.956 51 136.93
02:22:59.215 54 177.70
02:22:59.466 54 243.58
02:22:59.724 54 177.53
02:22:59.976 52 212.99
02:23:00.230 56 175.68
02:23:00.481 56 248.35
02:23:00.734 55 252.34
02:23:00.985 55 165.51
02:23:01.247 53 162.34
02:23:01.516 53 73.17
02:23:01.767 49 72.03
```

### The GPU watchdog of the GPU run: experiments/serving-runs/gpu_watchdog.sh

Run on 2026-10-12T03:06:44Z (UTC), exit status 0, in 0 s.

```console
$ cat experiments/serving-runs/gpu_watchdog.sh
#!/usr/bin/env bash
# GPU watchdog for the compose gpu run: samples temperature and power every 250 ms into a log,
# and stops the model server's container (only it) when the GPU reaches 78 C, or draws more than
# 230 W for 10 s in a row. Runs until killed. Usage: gpu_watchdog.sh <log file>
set -u
log=$1
over_since=""
stop_server() {
  echo "STOP $(date -u +%H:%M:%S) $1" >>"$log"
  docker stop codeforces-tags-model-server-gpu-1 >>"$log" 2>&1
}
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits -lms 250 |
  while IFS=', ' read -r temp power; do
    now=$(date +%s)
    echo "$(date -u +%H:%M:%S.%N | cut -c1-12) $temp $power" >>"$log"
    if [ "${temp%.*}" -ge 78 ]; then
      stop_server "temperature $temp C"
    fi
    if [ "${power%.*}" -gt 230 ]; then
      over_since=${over_since:-$now}
      if [ $((now - over_since)) -ge 10 ]; then
        stop_server "power above 230 W for 10 s ($power W)"
        over_since=""
      fi
    else
      over_since=""
    fi
  done
```

## Kubernetes on kind

### kind's node image

Run on 2026-10-11T22:37:10Z (UTC), exit status 0, in 16 s.

```console
$ docker pull -q kindest/node:v1.21.1@sha256:69860bda5563ac81e3c0057d654b5253219618a22ec3a346306239bba8cfa1a6 && docker image ls --digests kindest/node
docker.io/kindest/node:v1.21.1@sha256:69860bda5563ac81e3c0057d654b5253219618a22ec3a346306239bba8cfa1a6
REPOSITORY     TAG       DIGEST                                                                    IMAGE ID       CREATED       SIZE
kindest/node   <none>    sha256:69860bda5563ac81e3c0057d654b5253219618a22ec3a346306239bba8cfa1a6   32b8b755dee8   5 years ago   1.12GB
```

### The tfidf image of the Docker runs, loaded as it is (not rebuilt)

Run on 2026-10-11T22:37:32Z (UTC), exit status 0, in 0 s.

```console
$ git log -1 --format='HEAD %h' && docker image ls codeforces-tags/tfidf --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' && docker image inspect --format '{{index .Config.Labels "codeforces-tags.method"}} {{.Id}} {{.Created}}' codeforces-tags/tfidf:1029973 && grep -E '^  (code_dirty|git_commit):' build/serving/tfidf/model/MLmodel
HEAD b4762cb
codeforces-tags/tfidf:1029973 8af058113b18 906MB
codeforces-tags/tfidf:latest 8af058113b18 906MB
tfidf sha256:8af058113b1844c65dc7e680410156185e22ff898db0357d0f1c396a2777336a 2026-10-11T22:26:26.639313686Z
  code_dirty: false
  git_commit: 10299734e772a9df22c44a350d96f3b5a2541923
```

### Create the kind cluster

Run on 2026-10-11T22:37:38Z (UTC), exit status 0, in 53 s.

```console
$ kind create cluster --name codeforces-tags --wait 120s
Creating cluster "codeforces-tags" ...
 • Ensuring node image (kindest/node:v1.21.1) 🖼  ...
 ✓ Ensuring node image (kindest/node:v1.21.1) 🖼
 • Preparing nodes 📦   ...
 ✓ Preparing nodes 📦 
 • Writing configuration 📜  ...
 ✓ Writing configuration 📜
 • Starting control-plane 🕹️  ...
 ✓ Starting control-plane 🕹️
 • Installing CNI 🔌  ...
 ✓ Installing CNI 🔌
 • Installing StorageClass 💾  ...
 ✓ Installing StorageClass 💾
 • Waiting ≤ 2m0s for control-plane = Ready ⏳  ...
 ✓ Waiting ≤ 2m0s for control-plane = Ready ⏳
 • Ready after 20s 💚
Set kubectl context to "kind-codeforces-tags"
You can now use your cluster with:

kubectl cluster-info --context kind-codeforces-tags

Not sure what to do next? 😅  Check out https://kind.sigs.k8s.io/docs/user/quick-start/
```

### Load the tfidf image into the node

Run on 2026-10-11T22:38:40Z (UTC), exit status 0, in 42 s.

```console
$ kind load docker-image codeforces-tags/tfidf:1029973 --name codeforces-tags
Image: "codeforces-tags/tfidf:1029973" with ID "sha256:8af058113b1844c65dc7e680410156185e22ff898db0357d0f1c396a2777336a" not yet present on node "codeforces-tags-control-plane", loading...
```

### The overlay for the CPU, and what it renders

Run on 2026-10-11T22:39:36Z (UTC), exit status 0, in 3 s.

```console
$ uv run --offline python -m codeforces_tags.serving manifests --method tfidf --device cpu --tag 1029973 && cat build/k8s/tfidf-cpu/kustomization.yaml && kubectl kustomize build/k8s/tfidf-cpu
wrote <repo>/build/k8s/tfidf-cpu/kustomization.yaml: kubectl apply -k <repo>/build/k8s/tfidf-cpu
# Written by `python -m codeforces_tags.serving manifests`: write it again rather than edit it.
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../../deploy/k8s/overlays/cpu
namePrefix: tfidf-
labels:
- pairs:
    app.kubernetes.io/instance: 'tfidf'
  includeSelectors: true
images:
- name: codeforces-tags/model-server
  newName: 'codeforces-tags/tfidf'
  newTag: '1029973'
apiVersion: v1
kind: Service
metadata:
  labels:
    app.kubernetes.io/instance: tfidf
    app.kubernetes.io/name: model-server
  name: tfidf-model-server
spec:
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  selector:
    app.kubernetes.io/instance: tfidf
    app.kubernetes.io/name: model-server
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app.kubernetes.io/instance: tfidf
    app.kubernetes.io/name: model-server
  name: tfidf-model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/instance: tfidf
      app.kubernetes.io/name: model-server
  template:
    metadata:
      labels:
        app.kubernetes.io/instance: tfidf
        app.kubernetes.io/name: model-server
    spec:
      containers:
      - env:
        - name: CODEFORCES_DEVICE
          value: cpu
        envFrom:
        - secretRef:
            name: codeforces-tags-env
            optional: true
        image: codeforces-tags/tfidf:1029973
        imagePullPolicy: IfNotPresent
        livenessProbe:
          httpGet:
            path: /ping
            port: 8080
          periodSeconds: 30
          timeoutSeconds: 5
        name: model-server
        ports:
        - containerPort: 8080
        readinessProbe:
          httpGet:
            path: /ping
            port: 8080
          periodSeconds: 10
        resources:
          limits:
            memory: 8Gi
          requests:
            cpu: "1"
            memory: 4Gi
        startupProbe:
          failureThreshold: 30
          httpGet:
            path: /ping
            port: 8080
          periodSeconds: 10
```

### Apply the CPU overlay

Run on 2026-10-11T22:39:45Z (UTC), exit status 0, in 1 s.

```console
$ kubectl --context kind-codeforces-tags apply -k build/k8s/tfidf-cpu
service/tfidf-model-server created
deployment.apps/tfidf-model-server created
```

### Wait for the rollout

Run on 2026-10-11T22:39:46Z (UTC), exit status 0, in 30 s.

```console
$ kubectl --context kind-codeforces-tags rollout status deployment/tfidf-model-server --timeout=300s
Waiting for deployment "tfidf-model-server" rollout to finish: 0 of 1 updated replicas are available...
deployment "tfidf-model-server" successfully rolled out
```

### The Deployment, its pod and the Service

Run on 2026-10-11T22:40:16Z (UTC), exit status 0, in 0 s.

```console
$ kubectl --context kind-codeforces-tags get deployment,pods,service -o wide
NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE   CONTAINERS     IMAGES                          SELECTOR
deployment.apps/tfidf-model-server   1/1     1            1           30s   model-server   codeforces-tags/tfidf:1029973   app.kubernetes.io/instance=tfidf,app.kubernetes.io/name=model-server

NAME                                      READY   STATUS    RESTARTS   AGE   IP           NODE                            NOMINATED NODE   READINESS GATES
pod/tfidf-model-server-5849cdb9dd-995j9   1/1     Running   0          30s   10.244.0.5   codeforces-tags-control-plane   <none>           <none>

NAME                         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE    SELECTOR
service/kubernetes           ClusterIP   10.96.0.1       <none>        443/TCP    2m8s   <none>
service/tfidf-model-server   ClusterIP   10.96.104.196   <none>        8080/TCP   30s    app.kubernetes.io/instance=tfidf,app.kubernetes.io/name=model-server
```

### Smoke: tfidf on kind, through port-forward on local port 18080

Run on 2026-10-11T22:40:26Z (UTC), exit status 0, in 6 s.

```console
$ uv run --offline python -m codeforces_tags.serving smoke --url http://127.0.0.1:18080 --method tfidf
problem 13: 0.086 s, math: agrees
problem 14: 0.039 s, geometry: agrees
problem 15: 0.029 s, no tag: agrees
problem 18: 0.028 s, no tag: agrees
problem 29: 0.025 s, graphs: agrees
problem 37: 0.034 s, no tag: agrees
problem 39: 0.028 s, no tag: agrees
problem 60: 0.023 s, no tag: agrees
problem 62: 0.024 s, no tag: agrees
problem 64: 0.022 s, graphs: agrees
problem 68: 0.028 s, math: agrees
problem 69: 0.024 s, no tag: agrees
problem 71: 0.022 s, no tag: agrees
problem 74: 0.025 s, math: agrees
problem 75: 0.021 s, no tag: agrees
problem 79: 0.025 s, graphs: agrees
problem 89: 0.022 s, math: agrees
problem 95: 0.021 s, math, probabilities: agrees
problem 99: 0.024 s, no tag: agrees
problem 107: 0.020 s, strings, games: agrees
20 test problems, one request each: median 0.024 s, slowest 0.086 s; 0 differ from their frozen test rows
```

### The port-forward's log

Run on 2026-10-11T22:40:37Z (UTC), exit status 0, in 0 s.

```console
$ cat experiments/serving-runs/port-forward.log
Forwarding from 127.0.0.1:18080 -> 8080
Forwarding from [::1]:18080 -> 8080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
Handling connection for 18080
```

### The overlay for the GPU: what it adds

Run on 2026-10-11T22:40:44Z (UTC), exit status 0, in 3 s.

```console
$ uv run --offline python -m codeforces_tags.serving manifests --method tfidf --device gpu --tag 1029973 && kubectl kustomize build/k8s/tfidf-gpu | grep -n -E 'CODEFORCES_DEVICE|value: cuda|nvidia.com/gpu|NoSchedule|image:'
wrote <repo>/build/k8s/tfidf-gpu/kustomization.yaml: kubectl apply -k <repo>/build/k8s/tfidf-gpu
39:        - name: CODEFORCES_DEVICE
40:          value: cuda
45:        image: codeforces-tags/tfidf:1029973
64:            nvidia.com/gpu: 1
75:      - effect: NoSchedule
76:        key: nvidia.com/gpu
```

### The GPU overlay, server-side dry run

Run on 2026-10-11T22:40:47Z (UTC), exit status 0, in 1 s.

```console
$ kubectl --context kind-codeforces-tags apply -k build/k8s/tfidf-gpu --dry-run=server
service/tfidf-model-server unchanged (server dry run)
deployment.apps/tfidf-model-server configured (server dry run)
```

### Delete the cluster

Run on 2026-10-11T22:40:53Z (UTC), exit status 0, in 2 s.

```console
$ kind delete cluster --name codeforces-tags
Deleting cluster "codeforces-tags" ...
```

## The helper scripts

The records ran through small scripts of a gitignored folder, `experiments/serving-runs/`, written for these runs only. The runs of 2026-10-11 first sourced an environment file of that folder, which unset the API keys, so that no command could send a paid request, and set uv offline (`UV_OFFLINE=1`); the uv commands of the GPU run pass `--offline` themselves.

- `record.sh <id> <title> <command>` ran one command with bash (`pipefail`), and kept its title, the date it started (UTC), its exit status, its duration and its output, without terminal colour codes and with the repository's path written `<repo>`, as one Markdown record. `assemble_runs.py` put the records together, one part after the other, into the first version of this file.
- `start.sh <name> <command>` started a long-running command (a local model server, `serving mlflow-ui`, Streamlit, `kubectl port-forward`) in a session of its own and kept the id of its process group; `stop.sh <name>` sent that group SIGINT, then SIGKILL after 60 s.
- `serve_local.py --method M --port P` held a local model server open for `smoke` and the app: `serving.local`'s `start_local_server` on the packaged model `build/serving/<method>/model`, on the CPU, until Ctrl-C or SIGTERM.
- `download_sizes.py` printed what a download would fetch, before it was agreed, from the sizes that `uv.lock` records, and downloaded nothing. With `--requirements FILE`, it read every package that a `uv export` file pins for Linux x86_64 and CPython 3.12, as pip installs them in the image's build; with `--groups G ...`, the packages of dependency groups, with their dependencies, that the environment lacked. Each size is that of the package's wheel for Linux x86_64 and CPython 3.12, else of its sdist; `--top N` printed the N largest, then the count and the total.
- `links.py` printed the MLflow links that the Streamlit app's sidebar shows for three methods (`f2llm_late_fusion`, `tfidf` and `descode2tag-json`), from `serving.mlflow_links.method_links`.
- `app_check.py` ran `apps/streamlit_app.py` once with Streamlit's `AppTest`, without a browser, and printed its sidebar's texts and its buttons, or its exceptions.
- `gpu_watchdog.sh` sampled the GPU's temperature and power every 250 ms during the GPU run of compose, ready to stop the model server's container; its source is the last record of the Docker part.
