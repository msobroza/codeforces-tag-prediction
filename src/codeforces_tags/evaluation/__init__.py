"""Score predicted focus tags, with confidence intervals from a group bootstrap.

- ``report``: ``evaluate`` scores one set of predictions (per-tag precision, recall and F1,
  macro and micro F1, both confusion matrices, by slice of problems); ``compare`` gives the
  paired difference between two sets of predictions of the same problems;
  ``all_tag_macro_f1`` averages the F1 of many tags, to compare with the xCodeEval paper;
  ``report_text``, ``report_headline`` and ``save_report`` show an evaluation as text, JSON and
  CSV files (the command line's ``evaluate``).
- ``bootstrap``: the group bootstrap's resample weights and percentile intervals.
- ``predictions``: ``freeze_predictions`` saves a tagging method's predictions, scores and run
  record once, then only checks them; ``load_predictions``, ``load_scores``,
  ``load_run_record`` and ``load_per_problem`` read them back.
- ``timing``: the latency harness's timers (one problem at a time, a whole part, loading).
- ``contamination``: the self-identifying solutions, frozen once by
  ``python -m codeforces_tags.evaluation slices`` (``__main__``).
- ``calibration``: whether out-of-fold scores are probabilities: the four probabilities compared
  (``probabilities_by_method``, each calibrator fitted on the other folds), their Brier score,
  log loss and Brier skill with group-bootstrap intervals (``calibration_table``), and the Platt
  maps, which increase, so that mapped thresholds keep every cross-fitted prediction except where
  float rounding ties a score with its threshold, and which set each threshold beside half its F1
  (``platt_maps``, ``mapped_predictions``, ``lipton_table``).
"""
