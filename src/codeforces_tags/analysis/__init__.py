"""What the analysis notebooks compute and draw; nothing here is on the prediction path.

- ``eda``: every exploratory statistic (column profile, label counts and co-occurrence, lift,
  text and code lengths, odds ratios, distinctive terms).
- ``split_quality``: how good a dev/test split is, compared with random splits.
- ``near_duplicate_quality``: scores a duplicate detector against the judged pairs, and shows
  judged pairs for reading.
- ``error_analysis``: the problems a tag model gets wrong, read in notebook 04.
- ``pipeline_trace``: how the tag model's pipeline works, shown in notebook 04: the table that
  pairs each configuration field with the parameter it sets (``parameter_table``), and one
  problem followed through the fitted pipeline, from its raw fields to its tag scores
  (``trace_problem``).
- ``comparison``: tagging methods compared on the same problems: scores with intervals and
  paired differences (``method_scores``), and where two methods agree (``agreement``).
- ``llm_methods``: the tables of the zero-shot LLM methods' notebook, built from frozen files:
  each method's answer health (``answer_health``), tags per answer (``tags_per_answer``), all-tag
  macro F1 against the paper (``all_tag_scores``), cost per 1,000 problems
  (``cost_per_thousand``), and the problems where two methods disagree (``disagreements``).
- ``calibration``: whether a method's probabilities mean what they say: a reliability table per
  tag and its expected calibration error (``reliability_table``,
  ``expected_calibration_error``).
- ``prompt_search``: what a search over Jev's words did, for notebook 10, from its frozen record:
  each piece's word-level diff (``word_diff``, ``piece_changes``), the best candidate and the
  cost by iteration (``search_progress``), the chosen candidate's lineage (``lineage``), the
  refused proposals (``refusal_counts``), and how each tag's probabilities and predictions
  moved (``probability_shift``, ``flipped``).
- ``threshold_rules``: why a threshold rule matters, for notebook 11, from the frozen scores of
  a Jev method: a tag's F1 at each cut of Jev's 0.01 grid (``grid_cuts``, ``f1_curve``), the
  curve each fold's threshold is read on (``fold_curves``), the runs of cuts near a curve's
  maximum (``curve_peaks``), and Jev's probabilities Platt-mapped out of fold
  (``platt_probabilities``).
- ``calibrators``: notebook 11's comparison of calibrators, from the same scores: Jev's
  probabilities mapped out of fold by each calibrator (``calibrated_probabilities``), cut by each
  decision rule (``rule_predictions``) and ranked by ROC AUC and average precision
  (``ranking_table``).
- ``plotting``: the notebooks' plot style and shared plots, including each method's F1 and
  single-problem latency (``plot_method_f1``, ``plot_latency``) and the calibration's reliability
  diagrams (``plot_reliability``, ``plot_reliability_grid``), which ``tracking calibrate`` also
  logs, each tag's curve of a reliability table (``plot_reliability_table``), a search's best
  score and cost by iteration (``plot_search_progress``), and a tag's F1 curve with each
  threshold rule's thresholds (``plot_f1_curve``).

Only this package and ``tracking/``, which logs these figures, are meant to import matplotlib and
seaborn (the ``analysis`` dependency group). Ruff's ``TID253`` bans their module-level import
elsewhere, so predicting tags never loads them; ``per-file-ignores`` in ``pyproject.toml`` lists
the files exempt from it (CLAUDE.md, Code style).
"""
