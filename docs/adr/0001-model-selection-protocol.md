# Models are chosen on cross-fitted out-of-fold macro F1, and the test set is scored once

Every model is chosen on the dev set alone. Each Optuna trial fits the pipeline on four folds and scores the fifth. Its per-tag thresholds are cross-fitted: fold k's threshold is chosen on the out-of-fold scores of the other four folds, so the macro F1 the study maximises is not measured on the predictions the thresholds were fitted to. The chosen configuration is then refit on the whole dev set, with thresholds taken from all out-of-fold scores, and the test set is scored once, by that model. Error analysis reads out-of-fold predictions, never test ones.

## Considered options

- **Thresholds as Optuna dimensions**: spends most of the search budget on eight independent knobs.
- **Thresholds chosen on the pooled out-of-fold scores, then scored on them**: optimistic, since each threshold is fitted to the problems it is scored on, and a rare tag's to few positives (74 in dev for probabilities, 84 for games). For the chosen configuration, pooled thresholds give 0.669 macro F1 against 0.663 cross-fitted, a paired difference of +0.006 [0.002, 0.011] (notebook 04, section 4).

## Consequences

Each model reports one test score. Choosing between models, or changing one, because of its test score would make the test set a second dev set. A fixed reference model, which nothing is chosen from, may report its single test score beside the chosen one: notebook 04 scores the logistic regression of its first section this way. A method's frozen test predictions are logged per problem once, in one evaluation run per method; error analysis and per-problem comparisons read dev only.

Fine-tuned transformers (the ModernBERT methods) adapt this protocol to their training cost: they are tuned on one dev fold and ship as the bag of their fold models ([ADR 0005](0005-fine-tuned-model-protocol.md)).
