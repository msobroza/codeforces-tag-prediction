# Notebooks

Exploratory analysis, the split and the tagging methods of the tag-prediction task. The notebooks are committed with their outputs so they render on GitHub. Code shared across notebooks, or with the model, lives in [`src/codeforces_tags/`](../src/codeforces_tags/).

| Notebook | Content |
|---|---|
| [01_dataset_overview.ipynb](01_dataset_overview.ipynb) | Every variable, data quality, tag distribution and co-occurrence, and where the signal is in the statement and the code |
| [02_near_duplicates.ipynb](02_near_duplicates.ipynb) | Finding near-duplicate problems. Covers the data-leakage risk they pose for the split, why a single similarity threshold fails (with examples of misses and false links), and why the detector has two stages (with diagrams). Also covers where the LLM judges' verdicts come from, and how sure we are that no duplicate is missed |
| [03_train_test_split.ipynb](03_train_test_split.ipynb) | The dev/test split and its cross-validation folds. Covers why near-duplicate groups stay whole, why the problems with no focus tag are kept in their own stratum, and the method, including how iterative stratification works (with diagrams and pseudo code). Compares it with random splits over 50 seeds, then freezes it in `assets/split/split.csv` |
| [04_tfidf_baseline.ipynb](04_tfidf_baseline.ipynb) | The TF-IDF baseline. Covers a first look at every tag with one untuned logistic regression (the focus tags against the others), what the model reads (the statement normaliser, the code tokens, the sample I/O cues) and how its pipeline works (scikit-learn's diagram of it, a diagram of its cross-fitting, and the parameter each setting sets), the Optuna study that chose its configuration (with a diagram of the protocol), and its out-of-fold scores on the dev set with group bootstrap intervals, ablations of its feature blocks, both confusion matrices, its top features and one dev problem traced through the fitted pipeline. Then whether its scores can be read as probabilities: the first look's sigmoid and the tuned model's margins, min-max scaled, Platt-scaled or isotonic-calibrated on the other folds, compared out of fold in reliability diagrams and by Brier score, log loss and Brier skill with group bootstrap intervals, with the check that thresholds mapped through a Platt map change no prediction, and the mapped thresholds set beside half the F1 they reach. Then an error analysis on out-of-fold predictions, and last the test set: the tuned model's test predictions, frozen by `tuning freeze`, with the first look's logistic regression, scored once, as a reference |
| [05_zero_shot_llm.ipynb](05_zero_shot_llm.ipynb) | The xCodeEval paper's zero-shot Code2Tag (the solution alone) and DesCode2Tag (the statement, the notes and the solution), asked of `gpt-6-luna` in two output formats, free text and a strict JSON schema: the four zero-shot methods. Covers the setup beside the paper and the deviations from it, both prompts and a diagram of the pipeline, the health of the answers, the all-tag macro F1 beside the paper's, the focus tags against the TF-IDF baseline and the label prior, on dev (paired) and on test (scored once, side by side), free text against structured output (concluded on dev), slices by release era and without the self-identifying solutions, where TF-IDF and the LLM disagree, the latency and cost, and every dev problem in the MLflow UI: 13 screenshots of the store that `tracking evaluate` fills, with each problem's true tags, each method's predicted tags and, for the zero-shot methods, the tokens and cost of its call. Reads the methods' frozen files and those screenshots: it sends no request and needs no API key |
| [06_modernbert_lora.ipynb](06_modernbert_lora.ipynb) | The tagging method `modernbert_lora`: ModernBERT-large fine-tuned with LoRA on the statement and the solution (1,024 tokens). Covers the inputs and why they are cut at 1,024 tokens, the model and its training (fp16, gradient accumulation, weighted loss, no checkpoint), its Optuna study on one dev fold, its five fold models and their out-of-fold scores against the TF-IDF baseline and the label prior, then the test set, scored once by the bag of the five fold models with each single fold model and the references beside it, and the latency and memory of one encoder serving five adapters. Reads the method's frozen files |
| [07_modernbert_base_full.ipynb](07_modernbert_base_full.ipynb) | The tagging method `modernbert_base_full`: ModernBERT-base with every weight fine-tuned, no LoRA, on the statement and the solution (2,048 tokens), under the same protocol. Covers the search space of full fine-tuning, its tuning study on one dev fold, its five fold models compared out of fold, paired, with the TF-IDF baseline, the label prior and `modernbert_lora`, then the test set, scored once by their bag with each single fold model and the other methods as references, and the latency and memory of serving five full models from one model whose weights are swapped in. Reads the method's frozen files and `modernbert_lora`'s |
| [08_typesafe_jev.ipynb](08_typesafe_jev.ipynb) | The TypeSafe Jev methods, `jev-statement` and `jev-statement-code`, which ask TypeSafe's Jev model one question per focus tag, with the tag's definition, and read its probabilities. Covers what Jev is asked, what it costs and which models answered, the dev set out of fold with group bootstrap intervals against the label prior, TF-IDF and both ModernBERT methods, the statement against the statement and the solution, the calibration of the probabilities (reliability tables and curves, and a fixed 0.5 threshold against the cross-fitted ones), the release-era and self-identifying-solution slices, where the methods agree with ModernBERT and which problems they misread, then the test set, reported once with no choice made from it, and the latency of live requests. Reads the dataset and frozen files only: needs no key, no SDK and no GPU |
| [09_f2llm.ipynb](09_f2llm.ipynb) | The F2LLM embedding methods: the frozen embeddings of F2LLM-0.6B for the statement and the solution, and three tagging methods built on them (a linear probe, and late and early fusion with the TF-IDF baseline). Covers the input lengths and the share cut at 1,024 tokens, the out-of-fold results with group bootstrap intervals against the frozen TF-IDF method `tfidf` (also without the self-identifying solutions) and between the three methods, the paired difference per tag and per release era, the sensitivity to near-duplicates the lexical groups missed, the cross-fitted thresholds against pooled ones and their spread across the folds, the tuning grids and their plateau, late fusion's effective weights, the latency and eligibility of each method on the GPU (`latency.csv`) and on the CPU (`analysis/cpu/latency.csv`), and the load a one-shot call adds, and the test references (also without the test problems of a flagged dev-test pair). Reads frozen files only: no GPU and no `torch` |
| [10_jev_gepa.ipynb](10_jev_gepa.ipynb) | The searched Jev method `jev-statement-code-gepa`: GEPA's search over the words of `jev-statement-code` (its convention and its eight tag definitions), with `gpt-6-luna` as the reflection model, learning on dev folds 1-3 and choosing on fold 4 by a rule fixed beforehand ([ADR 0010](../docs/adr/0010-searched-jev-words.md)). Covers the checks of the frozen files and of the seed's score, what the search changed (each piece's word-level diff and length, and their effect on a request's tokens and cost), the search itself (the best selection-fold score by iteration with the accepted candidates, the cumulative cost of Jev and of the reflection model, the chosen candidate's lineage, the refusals by reason, and three or four MLflow screenshots of it), the held-out fold 0 against frozen `jev-statement-code` (macro F1, per-tag F1 and balanced Brier score, paired, with group bootstrap intervals), all of dev labelled in sample (each tag's mean probability against its base rate, before and after, and the new thresholds), the fold-0 problems whose tags flipped, the test set, reported once beside `jev-statement-code`, `tfidf` and `label_prior`, and the latency and cost of a request. Reads frozen files only: it sends no request and needs no key, no SDK and no GPU |
| [11_jev_platt.ipynb](11_jev_platt.ipynb) | The calibrated Jev twins `jev-statement-code-platt` and `jev-statement-code-gepa-platt`: the Jev probabilities of `jev-statement-code` and `jev-statement-code-gepa` under a threshold rule that reads calibrated probabilities (a Platt map on the logit, then Lipton's cut at half the best F1) instead of the F1 argmax ([ADR 0011](../docs/adr/0011-platt-lipton-threshold-rule.md)). Covers the checks of the frozen files (each twin's scores, `latency.csv` and `per_problem.csv` are its base's, the `test_informed` disclosure), Jev's reliability before and after the Platt map, per tag, a plain-words comparison of six calibrators (none, Platt on the logit or on the probability, temperature, beta and isotonic) with links to scikit-learn's documentation, why the F1 curve of the GEPA method's `probabilities` has several almost equal peaks (four runs within 0.01 of its maximum) and how the fold thresholds and the flipped problems of the two rules compare, the declared comparison of three rules and the nested choice, each twin against its base on dev (paired, with group bootstrap intervals) and the GEPA twin on its held-out fold 0, and the test set, reported once as a reference, since the rule was proposed after a test score was read. Reads frozen files only: it sends no request and needs no key, no SDK and no GPU |
| [12_model_serving.ipynb](12_model_serving.ipynb) | Serving a tagging method over HTTP with MLflow's own model server, in one Docker image per method that runs on local Docker, Kubernetes and SageMaker, on the CPU or the GPU ([ADR 0012](../docs/adr/0012-serving-tagging-methods.md)), with `f2llm_late_fusion`. Covers what is served and its frozen timings, MLflow's serving at a glance (figures from MLflow's documentation), the architecture, the MLflow model packaged from code from the committed method folder (its build context, `MLmodel`, requirements and lock) and its signature, a local model server on the CPU (`serving.local`) asked about three test-set problems against their frozen rows, a typed problem and a refused one, the server stopped, the image beside the Dockerfile MLflow writes, then the recorded runs of the Docker images and compose, a kind cluster and SageMaker's container contract, and the Streamlit client with its links to the MLflow experiment, the method's run and the prompt registry. It computes no metric and chooses nothing: the test-set problems are a demonstration |

## Running

Install the environment:

```bash
uv sync
```

Open the notebooks:

```bash
uv run jupyter lab notebooks/
```

The data is read from `code_classification_dataset/` at the repository root, a gitignored directory holding one `sample_<id>.json` per exercise. To use another location, set `CODEFORCES_DATA_DIR`.

After a change, re-execute the notebook in place so the committed outputs stay current:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_overview.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_near_duplicates.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_test_split.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_tfidf_baseline.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_zero_shot_llm.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/08_typesafe_jev.ipynb
```

Notebooks 06 and 07 read the frozen files of the ModernBERT methods, and count tokens with the pinned tokenizer, so they need the `finetuning` dependency group and the method's `fetch`:

```bash
uv sync --group finetuning
```

```bash
uv run --group finetuning jupyter nbconvert --to notebook --execute --inplace notebooks/06_modernbert_lora.ipynb
```

```bash
uv run --group finetuning jupyter nbconvert --to notebook --execute --inplace notebooks/07_modernbert_base_full.ipynb
```

Notebook 09 reads frozen files only, with no GPU and no `torch`: `assets/methods/f2llm_*/`, the frozen TF-IDF method `assets/methods/tfidf/` (checked against `assets/tfidf_baseline/`), `assets/methods/label_prior/`, [`assets/f2llm/`](../assets/f2llm/) and `assets/slices/self_identifying_solutions.csv`, besides the dataset for the labels. It checks that the TF-IDF reference rows of `assets/f2llm/` are the frozen method's rows. It takes under a minute. Its "What it shows" lines are computed by the cell above each one, so executing it again after a method has been frozen anew updates them:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_f2llm.ipynb
```

Notebook 10 reads the frozen files of `jev-statement-code-gepa` (the record of its search in `tuning/`, its answers files, predictions, scores, `per_problem.csv`, run record and `latency.csv`), those of `jev-statement-code`, `jev-statement`, `tfidf` and `label_prior`, the dataset for the labels, and the PNGs of `figures/mlflow/gepa/`. It sends no request and needs no key, no SDK and no GPU. It checks the searched method with `check_frozen_method`, and fails unless the seed's mean score on the selection fold, recomputed from the frozen `scores.csv` of `jev-statement-code`, is the one `tuning/params.json` records. Its prose quotes the frozen record and answers: another search is another method ([ADR 0010](../docs/adr/0010-searched-jev-words.md)), and after one, rewrite it:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/10_jev_gepa.ipynb
```

Notebook 11 reads the frozen files of the two calibrated twins (the record of each in `tuning/`, its predictions, scores, run record, method folder, `latency.csv` and `per_problem.csv`) and of their bases `jev-statement-code` and `jev-statement-code-gepa` (answers files included), those of `tfidf` and `label_prior`, and the dataset for the labels. It sends no request and needs no key, no SDK and no GPU. It checks the twins and their bases with `check_frozen_method` and the `test_informed` disclosure of each twin, and fails unless each twin's `scores.csv` equals its base's and its `latency.csv` and `per_problem.csv` are its base's, byte for byte. Its prose quotes the frozen record: another rule, or the same rule on another base, is another method ([ADR 0011](../docs/adr/0011-platt-lipton-threshold-rule.md)), and after one, rewrite it. The twins' own runs go to `tag-methods` under the rule of step 1 of [The MLflow screenshots](#the-mlflow-screenshots): they are evaluated on dev before the store's first test run, so a store that already holds test runs is replaced by a new one, built in that order with the twins frozen and evaluated on dev with the other methods (11 methods in all).

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/11_jev_platt.ipynb
```

Notebook 12 serves `f2llm_late_fusion` live, so it needs the dataset, the `f2llm` dependency group and the F2LLM weights in their cache (`python -m codeforces_tags.methods.f2llm fetch`), and a tree whose code, `deploy/Dockerfile` and method folder are committed: its first live cell runs `python -m codeforces_tags.serving package`, which refuses them otherwise. It writes the gitignored `build/serving/f2llm_late_fusion/` (with 1.2 GB of staged weights), serves it on the CPU with `serving.local` on a free port of 127.0.0.1, asks it about four problems and stops it, in about a minute. A cell that fails between the `serve` and `stop` cells leaves that server running, in a session of its own: `build/serving/f2llm_late_fusion/server.log` names its port, and a SIGTERM to its process group stops it. Its sections 10 to 12 quote the recorded runs of [`docs/serving-runs.md`](../docs/serving-runs.md), and its section 13 asks the MLflow UI that `uv run python -m codeforces_tags.serving mlflow-ui` serves for the served method's links when that UI answers, and otherwise prints the command that starts it. Re-executing it changes, beside the timestamps, its latencies, the server's port and the `MLmodel`'s commit, ids and times:

```bash
uv run --group f2llm jupyter nbconvert --to notebook --execute --inplace notebooks/12_model_serving.ipynb
```

Notebook 03 writes `assets/split/split.csv` only when it is missing. Otherwise it checks that the split it builds matches the file, and fails if not.

Notebook 04 reads the frozen results of the tuning study in [`assets/tfidf_baseline/`](../assets/tfidf_baseline/README.md), which `python -m codeforces_tags.tuning export` writes, and the method `tfidf` that `python -m codeforces_tags.tuning freeze` froze from them in `assets/methods/tfidf/`; it imports neither optuna nor mlflow. It cross-fits the chosen configuration again, and fails if the out-of-fold macro F1 differs from the frozen one, or if the thresholds differ from the frozen ones by more than `TUNING_EXPORT_TOLERANCE` (1e-9). It loads the frozen tag model instead of refitting it, and fails unless the tag model predicts the frozen test predictions exactly and their macro F1 is `TFIDF_TEST_MACRO_F1`. It fits 46 pipelines, so it takes a few minutes. The tuned model's test set was scored once by `tuning freeze`, as each tagging method's test set is scored by the method's own `freeze` command, and the notebook only reads those predictions; it scores the test set itself only for its reference, the first look's logistic regression, once, in its last results section. Its prose quotes the results of the frozen study (the chosen trial, its scores, the ablations, the plateau, the top features, the traced problem, its calibration). A new study that chooses another configuration gives a new model ([ADR 0004](../docs/adr/0004-tagging-method-framework.md)), which `tuning freeze` does not freeze as `tfidf`: it refuses another recipe than the one frozen in `assets/methods/tfidf/`, and `TFIDF_TEST_MACRO_F1` pins the frozen configuration's test score. Freezing a new configuration is a change of code, not a rerun; after it, rewrite that prose, and treat the new model's test score as a new model's single test score ([ADR 0001](../docs/adr/0001-model-selection-protocol.md)).

Notebook `05_zero_shot_llm` reads the frozen files of the four zero-shot methods in `assets/methods/` (`code2tag-text`, `code2tag-json`, `descode2tag-text` and `descode2tag-json`: their answers files, predictions, `per_problem.csv`, run records and `latency.csv`), those of `tfidf` and `label_prior`, the self-identifying slice and the dataset's true tags. It imports neither mlflow nor openai, sends no request, needs no API key, and takes about half a minute; re-executing it changes only its execution timestamps. It fails unless each answers file holds one answer per problem under the sha256 its run record froze, each method's frozen predictions are its answers parsed again, each method folder holds the package's template under its pinned prompt version, the comparison of free text with structured output reaches the conclusion committed before the first test request, and `tfidf`'s frozen macro F1 is notebook 04's, on dev and on test. The answers were asked once by `python -m codeforces_tags.llm run`, dev first (CLAUDE.md, LLM toolkit), and are never asked again ([ADR 0003](../docs/adr/0003-api-methods-are-fixed-reference-models.md)): another prompt is another method, with its own answers files. Its prose quotes the frozen answers. Its section 9, "Every problem in MLflow", shows the 13 PNGs of `figures/mlflow/` and imports no mlflow: it reads `config` and the PNG files, and its `mlflow-settings` cell fails unless the frozen files give the numbers the screenshots read. Re-executing the notebook does not retake them ([The MLflow screenshots](#the-mlflow-screenshots)).

Notebooks 06 and 07 train nothing: the commands of `codeforces_tags.methods.modernbert` (CLAUDE.md, Commands) froze the methods in `assets/methods/modernbert_lora/` and `assets/methods/modernbert_base_full/`, from the Optuna studies and fold models of the notebooks that first trained them ([ADR 0005](../docs/adr/0005-fine-tuned-model-protocol.md)). Each notebook reads its method's tuning results (`tuning/`), frozen predictions and scores, run record, latencies (`latency.csv` on the GPU, `analysis/cpu/latency.csv` on the CPU, and `analysis/notebook_latency.csv`, the first notebooks' single-model and bag timings) and each fold model's test scores (`analysis/fold_test_scores.csv`). It reads the TF-IDF baseline from the frozen method `tfidf` (`assets/methods/tfidf/`), as it reads `label_prior`'s, checks both with `check_frozen_method`, and checks the baseline's macro F1 against notebook 04's: out of fold against the tuning study's `out_of_fold.csv`, on the test set against `TFIDF_TEST_MACRO_F1`. Each bag is the method chosen before the test set was scored; the single fold models, the TF-IDF baseline and the label prior are references. Their prose quotes the frozen results: after a method is frozen again under a new name, rewrite it.

Notebook 08 reads the dataset and the frozen files of `label_prior`, the two TypeSafe Jev methods, `tfidf` and the two ModernBERT methods. It needs no API key, no SDK and no GPU, writes nothing to `assets/`, and takes about a minute. Its numbers are fixed by the frozen files (the group bootstrap is seeded), so its prose quotes them: after a new freeze, rewrite it. Re-executing it unchanged only updates the execution timestamps.

Re-executing a notebook whose code and data did not change only updates execution timestamps, except in notebook 03, whose two pandas Styler tables get a new random id (and a new memory address in their plain-text form) on every run. Discard these changes with `git checkout -- notebooks/<name>.ipynb`.

## Figures

Diagrams are written in Mermaid in `figures/*.mmd`, the single source of each diagram. Each one is rendered to a PNG next to it, and the notebooks display the PNG in a code cell. That way the image is saved in the notebook's outputs and shows everywhere: on GitHub, whose notebook preview does not render Mermaid, in JupyterLab, VS Code and HTML exports.

After editing a diagram, render the figures again, then re-execute the notebook that shows them. Rendering needs Node.js; the first run downloads `mermaid-cli` and a headless Chromium.

1. Render every diagram to its PNG:

   ```bash
   for f in notebooks/figures/*.mmd; do npx -y -p @mermaid-js/mermaid-cli@11.17.0 mmdc -i "$f" -o "${f%.mmd}.png" -b white -s 2; done
   ```

2. Re-execute the notebooks that show them (02, 03, 04, `05_zero_shot_llm`, 06, 07 and 12), so their saved outputs show the new images:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_near_duplicates.ipynb
   ```

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_test_split.ipynb
   ```

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_tfidf_baseline.ipynb
   ```

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_zero_shot_llm.ipynb
   ```

   ```bash
   uv run --group finetuning jupyter nbconvert --to notebook --execute --inplace notebooks/06_modernbert_lora.ipynb
   ```

   ```bash
   uv run --group finetuning jupyter nbconvert --to notebook --execute --inplace notebooks/07_modernbert_base_full.ipynb
   ```

   ```bash
   uv run --group f2llm jupyter nbconvert --to notebook --execute --inplace notebooks/12_model_serving.ipynb
   ```

Rendering an unchanged diagram gives a byte-identical PNG, so `git status` shows only the figures you edited.

One figure is not Mermaid. `figures/tfidf_estimator.png` is scikit-learn's own diagram of the TF-IDF pipeline (`estimator_html_repr` of `build_pipeline` for the frozen `params.json`), with every step box open; notebook 04 shows it instead of the estimator, whose HTML output weighs 136-170 KB and holds memory addresses that change on every run. `figures/estimator_diagram.py` renders it in headless Chrome, driven by playwright, which only the `figures` dependency group installs (CI does not). Run it after a change to `build_pipeline`, `ModelParams` or `params.json`, then re-execute notebook 04:

```bash
uv run --group figures python notebooks/figures/estimator_diagram.py
```

It uses the system Chrome. Without one, run `uv run --group figures playwright install chromium` once, and it uses that Chromium. It aborts every http(s) request, so it needs no network and the PNG holds no remote asset, and it leaves the "Parameters" tables of the boxes closed, since they show the address of every function the pipeline holds. Rendering again gives a byte-identical PNG on one machine, not necessarily on another (fonts and the Chrome version differ).

Three figures are not ours. `figures/mlflow_docs/` holds three figures of MLflow 3.16.1's documentation that notebook 12 shows, copied once and unchanged: MLflow's deployment overview, its models-from-code diagram and its signature-enforcement diagram. MLflow's documentation is under the Apache License 2.0: `figures/mlflow_docs/LICENSE-mlflow.txt` is that licence, and `figures/mlflow_docs/README.md` names each figure's source and notes that the overview's "Flask" is out of date.

### The MLflow screenshots

The 13 screenshots of notebook 05's section 9, `figures/mlflow/*.png`, are not Mermaid either: `figures/mlflow_ui.py` takes them from the MLflow 3.16.1 UI, which shows every problem of the dev set with the predictions of the six methods that notebook 05 compares. They are of dev only ([ADR 0001](../docs/adr/0001-model-selection-protocol.md)): no view opens the test dataset, a test run or a test trace. The UI's LLM features (Assistant, Analyze, Detect Issues) are never opened, nor is any control that writes. Retaking them needs the local MLflow store, the `figures` dependency group and Chrome, as the estimator diagram does, and takes about two hours, nearly all of it the 12 evaluation runs of the store.

1. Rebuild the store (`experiments/mlflow.db` and `experiments/mlartifacts/`, gitignored) from committed files. No request is sent and no API key is needed: run these without `OPENAI_API_KEY` in the environment, so that a request would fail instead of costing money. Start from a store of its own: a fresh worktree has none, and in a checkout that holds one, move it aside while no MLflow server or command runs on it: `mkdir experiments/old-store && mv experiments/mlflow.db* experiments/mlartifacts experiments/old-store/`. `mlflow.db*` takes the `-wal` and `-shm` files with `mlflow.db`: the store is in WAL mode, so `mlflow.db` alone would lose what its `-wal` file holds, and a `-wal` file left behind would be replayed into the new store. The rest of `experiments/` is journals and caches, which can stay. Never copy another worktree's store, since it records the absolute path of its artifacts. The commands read the dataset of `CODEFORCES_DATA_DIR`, check the frozen files they read and are idempotent, so running one again adds nothing. First the six methods' runs, their prompts and timings, and the evaluation datasets (about 4 minutes):

   ```bash
   uv run python -m codeforces_tags.methods.label_prior freeze
   uv run python -m codeforces_tags.tuning freeze
   for method in code2tag-text code2tag-json descode2tag-text descode2tag-json; do
     uv run --group llm python -m codeforces_tags.llm score --method "$method"
   done
   for method in label_prior tfidf; do
     uv run python -m codeforces_tags.tracking latency --method "$method"
   done
   for method in code2tag-text code2tag-json descode2tag-text descode2tag-json; do
     uv run --group llm python -m codeforces_tags.tracking latency --method "$method"
   done
   uv run python -m codeforces_tags.tracking calibrate
   uv run python -m codeforces_tags.tracking datasets
   ```

   Then the 12 evaluation runs: every dev run, a 2-minute pause, then every test run, in the order that puts `descode2tag-json`, the method the views compare, last (about 95 minutes: a dev run takes 11 to 14 minutes, a test run about 3, and the store grows to about 600 MB):

   ```bash
   for method in label_prior tfidf code2tag-text code2tag-json descode2tag-text descode2tag-json; do
     uv run python -m codeforces_tags.tracking evaluate --part dev --method "$method"
   done
   sleep 120
   for method in label_prior tfidf code2tag-text code2tag-json descode2tag-text descode2tag-json; do
     uv run python -m codeforces_tags.tracking evaluate --part test --method "$method"
   done
   ```

   `--method` names each one: a bare `tracking evaluate` takes every frozen method, and refuses before its first run when one of them, here the F2LLM, Jev and ModernBERT methods, has no logged model in the store. The Overview and the Traces page show the time window of the dev runs, in whole minutes, and the script refuses a store where a test run falls in it: hence the order and the pause. A run that stops is discarded with `tracking discard --run <run id>`, and the loop started again, which checks the complete runs and goes on. A method that joins the store later is logged by its own `freeze` (CLAUDE.md, Commands; for example `uv run --group llm-typesafe python -m codeforces_tags.methods.typesafe_jev freeze --method jev-statement-code`) and evaluated on dev before the store's first test run. Its runs then appear in the views, and the captions of notebook 05, which count six methods and 23,928 dev traces, need rewriting. A store that already holds test runs is replaced by a new one, built in the order above.

2. Take the screenshots:

   ```bash
   unshare -rn sh -c 'ip link set lo up && env -u OPENAI_API_KEY -u TYPESAFE_API_KEY uv run --offline --group figures python notebooks/figures/mlflow_ui.py'
   ```

   This is the run that took the committed PNGs. Install the `figures` group once beforehand, outside the namespace (`uv sync --group figures`): inside it there is no network, so `uv run` needs `--offline` and finds the group installed. `unshare -rn` (util-linux) opens a network namespace whose only interface is the loopback, which `ip link set lo up` (iproute2) brings up, so that nothing the server or Chrome do can leave the machine: the script aborts any request that a page or a dedicated worker makes to another address and removes shared workers, whose requests it could not see, but it only records a request its routes do not see (a redirect's target, already sent), and it cannot see the server's own requests or Chrome's. Without a namespace, run the script alone, on the machine's network. It starts `mlflow server` on 127.0.0.1:5517 (`MLFLOW_UI_PORT`; a port in use is refused) with one worker, no job runner and no credential or proxy variable (API keys, tokens, cloud keys, `*_PROXY`), reads it through no proxy, checks that it serves this store, and opens the 13 views in headless Chrome. It uses the system Chrome, else playwright's Chromium (`uv run --group figures playwright install chromium`). It takes about 2 minutes, and prints `<n> requests, all to 127.0.0.1:5517, and 0 inline`, the weight of the PNGs (it quantizes them to 256 colours when together they pass 2.5 MB) and `Wrote 13 PNGs to <folder>`. `--out DIR` writes the PNGs elsewhere, to compare a retake with the committed ones, and `--port N` uses another port.

   The exit status is 0 when the 13 PNGs were replaced, 1 on a refusal or a failure (a port in use; a store that does not exist, or lacks what a view opens, or holds an incomplete evaluation run or a second dev run of a method; a test run that meets the dev window; another server answering; a control missing from the UI; a capture that misses a file) and 2 when a request left the server, whose URLs it prints, even if a view then failed (a page can wait for what was aborted). On 1 and on 2 no PNG changes: the views go to a temporary folder beside `figures/mlflow/`, and move in only when all 13 succeeded and no request left the server. The server is stopped in every case.

   The PNGs are retaken, not reproduced: the relative times (`3 hours ago`), the order of the tag pills and the run colours differ from one run to the next, and a quantized PNG's palette follows its pixels. Commit a retake only when a view changed (a new MLflow version, a new method in the store).

3. Re-execute notebook 05, which embeds the PNGs (it weighs about 3.3 MB), and read its section 9 against them:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_zero_shot_llm.ipynb
   ```

   Its `mlflow-settings` cell fails unless the frozen files give the numbers the screenshots read, and `tests/figures/test_mlflow_ui.py` fails unless the views of the script, the PNGs of the folder and the images of the notebook are the same 13, in the same order.

### The search's MLflow screenshots

The three or four screenshots of notebook 10's search section, `figures/mlflow/gepa/*.png` (`config.MLFLOW_SCREEN_GEPA_FOLDER`), come from the same script with `--set gepa`. They show the search over Jev's words, which reads dev folds 1-4 only ([ADR 0010](../docs/adr/0010-searched-jev-words.md)), from its own experiment, `jev-gepa`, and nothing of `tag-methods`:

- `01_gepa_search_chart`: the parent run `search-jev-statement-code-gepa`'s charts of `selection_best_score` and of the costs, by iteration;
- `02_gepa_candidates`: its nested runs, one per candidate scored on the selection fold, sorted by score;
- `03_gepa_prompt_compare`: the prompt `jev-gepa-<tag>` of the tag `config.MLFLOW_SCREEN_GEPA_TAG`, the tag whose words changed most, comparing version 1 (the seed's words) with the chosen version;
- `04_gepa_convention_compare`: the prompt `jev-gepa-convention` compared the same way, only when the chosen words changed the convention.

They need no evaluation run and no paid call: the search's record is committed, and `log-search` replays it into the store.

1. Replay the search into the store (it adds nothing the store already holds, so running it again is harmless):

   ```bash
   uv run python -m codeforces_tags.methods.typesafe_jev log-search --method jev-statement-code-gepa
   ```

2. Take the screenshots, in the same network namespace as the 13 views:

   ```bash
   unshare -rn sh -c 'ip link set lo up && env -u OPENAI_API_KEY -u TYPESAFE_API_KEY uv run --offline --group figures python notebooks/figures/mlflow_ui.py --set gepa'
   ```

   The script talks to 127.0.0.1 only, as for the 13 views, and replaces the PNGs of `figures/mlflow/gepa/` only when every view of the set succeeded. `--set methods`, the default, takes the 13 views of notebook 05, unchanged.

3. Re-execute notebook 10, which embeds them:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/10_jev_gepa.ipynb
   ```

The searched method's own runs (its logged model, logged by its `freeze`, and its evaluation runs) go to `tag-methods` under the rule of step 1 above: it is evaluated on dev before the store's first test run, so a store that already holds test runs is replaced by a new one, built in step 1's order with `jev-statement-code-gepa` frozen and evaluated on dev with the other methods. They change none of the search's views.

### The Streamlit screenshots

The two screenshots of notebook 12's section 13, `figures/streamlit/test_problem.png` and `figures/streamlit/new_problem.png`, come from `figures/streamlit_app.py`. It starts the Streamlit app (`apps/streamlit_app.py`) in Streamlit's headless server on a free port of 127.0.0.1, against a model server and an MLflow UI that already run, opens it in headless Chrome, on a page 1700 px wide, and takes two views, each cropped to its content: a test-set problem, and a typed problem (the totient problem of notebook 12, without its sample pairs), each once the model server has answered and the sidebar shows its links to the `tag-methods` experiment, the method's run and the prompt registry. The test-set problem is chosen by a fixed rule that never looks at a prediction (`first_tagged_problem`): the first problem, in the picker's order, with at least one true focus tag, which is problem 14 with the frozen split. The served method must be a score method, since each view waits for the chart of the scores, and the store must hold its run: the capture refuses a sidebar that says "not logged in this store". The committed screenshots were taken against a local model server of `f2llm_late_fusion` on the CPU (`serving.local`, on port 18081) and the store's MLflow UI on port 15000, named by `--url` and `--mlflow-url`, so the round trips they show are CPU ones.

1. Serve `f2llm_late_fusion` on 127.0.0.1:8080 (for instance its image under compose: `METHOD=f2llm_late_fusion docker compose -f deploy/docker-compose.yml --profile cpu up -d --wait`) and the store's MLflow UI on 127.0.0.1:5000 (`uv run python -m codeforces_tags.serving mlflow-ui`, in another shell), with the dataset in `CODEFORCES_DATA_DIR`.
2. Take the screenshots:

   ```bash
   uv run --group figures --group app python notebooks/figures/streamlit_app.py
   ```

   `--url` and `--mlflow-url` name other servers (by default `$CODEFORCES_SERVING_URL`, else http://127.0.0.1:8080, and `$CODEFORCES_MLFLOW_UI_URL`, else http://127.0.0.1:5000); both must be http URLs of 127.0.0.1, and both must answer before the app starts. The app's server gets no credential, and its proxy variables point at a closed port of 127.0.0.1, so that nothing it asks of another host leaves the machine; the browser aborts any request that leaves the app's server. It prints the number of requests and the weight of each PNG, then `Wrote 2 PNGs to <folder>`. The exit status is 0 when both PNGs were replaced, 1 when the capture missed a file and 2 when a request left the app's server; on a failure no PNG changes, and the app's server is stopped in every case.
3. Stop both servers, then re-execute notebook 12, which embeds the PNGs.

## Conventions

- **Loading**: `codeforces_tags.data.loading.load_dataset()` gives the preprocessed frame. `read_samples()` gives the raw values as stored, and `codeforces_tags.preprocessing.pipeline.preprocess()` turns them into the preprocessed frame.
- **Shared code**: notebooks define no functions. Exploratory statistics go in `codeforces_tags/analysis/eda.py`, plots in `analysis/plotting.py`, measures of a split in `analysis/split_quality.py`, measures of the near-duplicate detector in `analysis/near_duplicate_quality.py`, the listing of a model's errors in `analysis/error_analysis.py`, the scores of several tagging methods and where they agree in `analysis/comparison.py`, the tables of the zero-shot LLM methods (their answers' health, the all-tag macro F1, the cost per 1,000 problems, their disagreements) in `analysis/llm_methods.py`, the tag model's parameter table and the trace of one problem through it in `analysis/pipeline_trace.py`, the F1 curves of the threshold rules, their peaks and the cross-fitted Platt probabilities in `analysis/threshold_rules.py`, the comparison of calibrators in `analysis/calibrators.py`, near-duplicate detection in `near_duplicates/`, the split itself in `splitting.py`, the tag model (features, thresholds, training) in `model/`, its metrics and group bootstrap intervals in `evaluation/`, the tuning study and the reading of its frozen results in `tuning/`, and anything the model also needs in `data/` or `preprocessing/`, each with tests in `tests/`.
- **Settings**: thresholds, tag lists, ablations and plot colours shared with the package come from `codeforces_tags.config`, not from literals in the notebook.
- **Naming**: notebooks are numbered in reading order, `NN_topic.ipynb`.
