# TF-IDF baseline: tuning results

The frozen results of the Optuna study that chose the configuration of the TF-IDF baseline. `python -m codeforces_tags.tuning export` writes the five data files from the study. `load_tuning_results()` in `codeforces_tags.tuning.results` reads them back, so [`notebooks/04_tfidf_baseline.ipynb`](../../notebooks/04_tfidf_baseline.ipynb) uses them without importing optuna or mlflow. `python -m codeforces_tags.tuning freeze` fits the chosen configuration from them and freezes it as the tagging method `tfidf`, in [`assets/methods/tfidf/`](../methods/tfidf/).

The study itself, its Optuna journal and its MLflow store, stays in the gitignored `experiments/` folder. These files are its only committed record.

## The study

- **Dev set only.** Every trial is scored on the dev set of the frozen split ([`assets/split/split.csv`](../split/split.csv)), over its five folds. The test set is never read.
- **Objective.** A trial draws one configuration, fits it on four folds and scores the fifth, for each fold in turn. Its per-tag thresholds are cross-fitted: fold k's threshold is chosen on the out-of-fold scores of the other four folds. The trial's value is the macro F1 of these out-of-fold predictions ([ADR 0001](../../docs/adr/0001-model-selection-protocol.md)).
- **Sampler.** Multivariate TPE (`group=True`, `constant_liar=True`), with 30 random startup trials (`config.TUNING_STARTUP_TRIALS`), until the study has 200 complete trials (`config.TUNING_TRIALS`).
- **Workers.** 4 worker processes (`config.TUNING_WORKERS`) share the journal, and worker i samples with seed `config.TUNING_SEED + i`. A worker finishes the trial it is running when the study reaches 200 complete trials, so the study can end with up to 203.
- **Choice.** The configuration of the trial with the highest value. `export` cross-fits it again on the dev set, which gives the out-of-fold scores and final thresholds saved here. It fits no tag model: the tag model is a tagging method, logged to MLflow through `log_method` ([ADR 0002](../../docs/adr/0002-method-folders-and-mlflow-wrapper.md)).

### Search space

`suggest_params()` in `codeforces_tags.tuning.search` draws a configuration, with the ranges of `config.TUNING_SPACE`. The statement block is always kept, with weight 1, and so is the code block.

| Parameter | Values, as run |
|---|---|
| `classifier` | `logistic_regression` (liblinear) or `linear_svm` |
| `c_logistic_regression`, `c_linear_svm` | C of the classifier drawn, log scale: [0.01, 30] and [0.001, 3] |
| `class_weight` | `none`, or `balanced`: in each tag's model, a problem weighs n / (2 · n_class) |
| `include_notes` | Whether the notes are appended to the statement |
| `statement_ngram_max`, `statement_min_df`, `statement_max_df`, `statement_sublinear_tf` | Statement TF-IDF: n-grams up to 1 to 3 words, minimum document count 1 to 5, maximum document frequency [0.5, 1.0], sublinear TF |
| `code_weight` | Weight of the code block, [0.1, 1.5] |
| `code_ngram_max`, `code_min_df`, `code_sublinear_tf` | Code TF-IDF: n-grams up to 1 to 3 tokens, minimum document count 2 to 10, sublinear TF |
| `use_cues`, `cue_weight` | Whether the five sample I/O cues are kept, and then their weight, [0.05, 1.0] |

## Files

| File | Content |
|---|---|
| `params.json` | The chosen configuration, `ModelParams.to_dict()`: the fields of `ModelParams`, with `c` the C of the classifier chosen, `cue_weight` 0 when the cue block is dropped, and `blocks` the feature blocks kept |
| `thresholds.json` | Its per-tag thresholds, chosen on all its out-of-fold scores: a tag is predicted when the model's `decision_function` score reaches it. `Infinity` would mean that no dev problem has the tag |
| `trials.csv` | One row per trial of the study, with the columns below |
| `importances.csv` | `parameter`, `importance`: the fANOVA importance of each parameter for the macro F1 (Optuna's `FanovaImportanceEvaluator`, seeded with `config.TUNING_SEED`), over the complete trials: each parameter's main effect, normalised so that the importances sum to 1 (the interactions between parameters are left out), highest first. Only the parameters every trial draws are measured: the two C parameters and `cue_weight` are left out |
| `out_of_fold.csv` | `metric`, `f1`: the chosen configuration's out-of-fold F1 with cross-fitted thresholds, in rows `macro_f1`, `micro_f1`, then one per focus tag. Its `macro_f1` equals the best `value` of `trials.csv`: `export` saves nothing unless its re-run of the best configuration scores that value, to within `config.TUNING_EXPORT_TOLERANCE` (1e-9) |

### Columns of `trials.csv`

| Column | Meaning |
|---|---|
| `number` | Trial number, in the order the trials started |
| `state` | `COMPLETE`; `FAIL` when the trial raised or was interrupted (Ctrl-C); `RUNNING` if exported while it ran, or after its worker was killed (see [Resume an interrupted study](#resume-an-interrupted-study)) |
| `value` | Out-of-fold macro F1, with cross-fitted thresholds; empty unless the trial is complete |
| `duration_s` | Wall time of the trial in its worker, in seconds |
| `params_<name>` | The value drawn for each parameter of the search space; empty when the trial did not draw it (the C of the other classifier, `cue_weight` without cues) |
| `micro_f1` | Out-of-fold micro F1 |
| `f1/<tag>` | Out-of-fold F1 of each focus tag |

## Reproducibility

- **The study is not bit-reproducible.** The four workers share one journal, and through the constant liar each one samples knowing which trials the others are running. Which configurations are drawn therefore depends on the order in which trials finish, that is, on timing. A rerun explores other configurations and may choose another one. These files are the record of the run.
- **A trial is.** The score of a configuration is deterministic: the TF-IDF blocks have no randomness and liblinear is seeded (`config.MODEL_SEED`). `export` on the same journal writes the same files on the same machine. Elsewhere, the thresholds agree to within 1e-9: the number of BLAS threads changes the scores by rounding errors (within 1e-9, as `tests/model/test_training.py` checks), and the thresholds come from the scores. `fit_tag_model` with `params.json` gives every row of `out_of_fold.csv` again, exactly, and the thresholds of `thresholds.json` to within 1e-9: `tuning freeze` writes nothing otherwise, `tuning verify` checks it again on the frozen method, and notebook 04 checks the `macro_f1`.

## Reproduce

The commands run from the repository root. They read the dataset from `code_classification_dataset/`, or from the folder in `$CODEFORCES_DATA_DIR`.

### Run the study

`experiments/` must not hold an earlier journal or MLflow store, or the study would continue it.

```bash
uv sync --locked
mkdir -p experiments
nohup uv run python -m codeforces_tags.tuning run --trials 200 --workers 4 >> experiments/tuning.log 2>&1 &
```

- **Threads.** Each worker runs BLAS on one thread (`config.TUNING_BLAS_THREADS`). Otherwise the BLAS calls of liblinear start one thread per CPU in each of the four workers, and the threads compete: on 12 CPUs, the same 15 trials took twice as long.
- **Progress.** `tail -n 5 experiments/tuning.log` shows the last trials to end (`tail -f` follows them), and `grep -c "finished with value" experiments/tuning.log` counts the complete trials. `uv run optuna best-trial --storage experiments/optuna_journal.jsonl --storage-class JournalFileBackend --study-name tfidf-baseline -f yaml` shows the best trial so far.
- **End.** `run` ends with a line such as `200 complete trials, best macro F1 ... (trial ...); MLflow run ...`, after logging every complete trial to MLflow. It exits with status 1, and logs nothing, if a worker failed or no trial is complete.

### Resume an interrupted study

The journal keeps every finished trial, so the same `run` command continues an interrupted study until it has 200 complete trials. On a complete study, it only logs the trials MLflow lacks.

A worker stopped during a trial (`kill`, power loss) leaves that trial `RUNNING` in the journal. Before resuming, stop any process of the study still alive (`pkill -f '^[^ ]*python[^ ]* -m codeforces_tags.tuning'`), then mark these trials as failed:

```bash
uv run python - <<'EOF'
from optuna.trial import TrialState

from codeforces_tags.tuning.search import create_study

study = create_study()
for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
    study.tell(trial.number, state=TrialState.FAIL)
    print(f"trial {trial.number} was left running: marked failed")
EOF
```

A resumed worker restarts its seeded random stream, so its first trials can repeat configurations of the interrupted run.

### Freeze the results

```bash
uv run python -m codeforces_tags.tuning export
```

It writes the five files here, and logs the study to MLflow (its run and the trials not logged yet), as `run` does. To browse the study and its trials, open http://127.0.0.1:5000 after:

```bash
uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```

### Freeze the tag model as the method `tfidf`

```bash
uv run python -m codeforces_tags.tuning freeze
uv run python -m codeforces_tags.tuning verify
uv run python -m codeforces_tags.tracking latency --method tfidf
uv run python -m codeforces_tags.tracking calibrate
```

`freeze` never re-runs the study. It reads `params.json`, fits the tag model on the dev set (`fit_tag_model`: every dev problem scored out of fold, the thresholds chosen on those scores, then the pipeline refit on every dev problem) and, before writing anything, checks that it reproduces the files here: every row of `out_of_fold.csv` exactly, the thresholds within `config.TUNING_EXPORT_TOLERANCE` (1e-9) of `thresholds.json`, and a test macro F1 equal to `config.TFIDF_TEST_MACRO_F1`, notebook 04's test score before the tag model became a tagging method. It then writes [`assets/methods/tfidf/`](../methods/tfidf/) once: `predictions.csv` and `scores.csv` (the dev rows out of fold, never the refit's in-sample ones, and the refit's test rows), `run_record.json` (its recipe hashes `params.json`, `thresholds.json`, the recorded settings and the focus tags) and the method folder `model/` (`model.skops`, the fitted `CrossFittedTagThresholds` saved by skops and compressed, `tag_model.json` and `method.json`), which must predict every dev and test problem as the fitted tag model does and score them within 1e-9 of it; a `model/` left without the frozen files beside it may come from another fit, and is refused. It refuses uncommitted code unless given `--allow-dirty`, which the run record records, and logs the method to the `tag-methods` experiment of the MLflow store, not to the study's `tfidf-baseline`. A second run rewrites nothing: it derives everything again, checks it against the frozen files and logs only what the store lacks, so it also rebuilds an empty store.

A rerun of the study may choose another configuration (see [Reproducibility](#reproducibility)). That is a new model ([ADR 0004](../../docs/adr/0004-tagging-method-framework.md)), which `freeze` does not freeze as `tfidf`. While `assets/methods/tfidf/` holds the frozen run record, `freeze` refuses another recipe before it fits anything; without it, `freeze` checks the study first, then refuses a test macro F1 other than `config.TFIDF_TEST_MACRO_F1`, the frozen configuration's, and does not print it, so that no configuration's test score is read before it is frozen ([ADR 0001](../../docs/adr/0001-model-selection-protocol.md)). Freezing a new configuration is a change of code: a new method, or a reviewed replacement of `tfidf` and of `TFIDF_TEST_MACRO_F1`.

`verify` writes nothing. It checks the frozen files without fitting (the dev predictions and the thresholds from `scores.csv`, every row of `out_of_fold.csv` and the test macro F1 from `predictions.csv`, the test rows from the method folder), then fits the tag model again as `freeze` does and compares: predictions exactly, scores within 1e-9. `tracking latency` times the frozen method once on 50 dev problems and writes its `latency.csv`. All three read the dataset of `$CODEFORCES_DATA_DIR` (or `--data`); `freeze` and `verify` take about 45 s each.

`tracking calibrate` asks, as notebook 04's section 5 does, whether the frozen method's out-of-fold scores are probabilities, and logs the answer to the method's MLflow run: the calibration table as `calibration/calibration_table.csv`, one metric per estimate of the table (108) and two reliability diagrams. It computes everything from the dev problems only (it reads and hashes the whole dataset, then keeps the dev rows of the split `tfidf` was frozen with, and refuses another split), takes the scores from the method folder without refitting it, cross-fits the first look on the 8 focus tags, and writes nothing under `assets/` (a few minutes, the dataset read like the other three). It recomputes everything at every run but never updates a metric or a file that the run already holds: a second run logs nothing new, and after a change of code the run keeps its old values. To log the new ones, rebuild the run in a fresh store: give `freeze`, `tracking latency` and `tracking calibrate` the same new `--tracking-uri` and `--artifact-dir`.

### Record the run

The run record below comes from the journal, the git commit and the installed libraries:

```bash
uv run python - >> assets/tfidf_baseline/README.md <<'EOF'
import os
import platform
import subprocess
from importlib.metadata import version

from codeforces_tags.tuning.search import create_study

study = create_study()
trials = study.trials_dataframe(
    attrs=("number", "state", "datetime_start", "datetime_complete", "duration")
)
start, end = trials["datetime_start"].min(), trials["datetime_complete"].max()
complete = trials[trials["state"] == "COMPLETE"]
states = trials["state"].value_counts()
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
rows = {
    "Run": f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}, "
    f"{(end - start).total_seconds() / 60:.0f} min of wall time",
    "Code": f"commit `{commit}`",
    "Trials": ", ".join(f"{count} {state.lower()}" for state, count in states.items()),
    "Best trial": f"{study.best_trial.number}: out-of-fold macro F1 {study.best_value:.4f}",
    "Mean trial duration": f"{complete['duration'].mean().total_seconds():.0f} s in one worker",
    "Machine": f"{platform.system()}, {os.cpu_count()} CPUs",
    "Python": platform.python_version(),
}
libraries = ("numpy", "pandas", "scipy", "scikit-learn", "optuna", "mlflow", "skops")
rows |= {library: version(library) for library in libraries}
print("\n## Run record\n")
print("| Item | Value |")
print("|---|---|")
for item, value in rows.items():
    print(f"| {item} | {value} |")
EOF
```

## Run record

| Item | Value |
|---|---|
| Run | 2026-10-06 01:49 to 2026-10-06 02:24, 35 min of wall time |
| Code | commit `e9a207d` |
| Trials | 203 complete |
| Best trial | 185: out-of-fold macro F1 0.6629 |
| Mean trial duration | 41 s in one worker |
| Machine | Linux, 12 CPUs |
| Python | 3.12.12 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| optuna | 5.0.0 |
| mlflow | 3.16.1 |
| skops | 0.16.0 |

The machine also ran another project's jobs during the study (load average about 28 on 12 CPUs), so the wall time and the trial durations are higher than they would be on an idle machine.
