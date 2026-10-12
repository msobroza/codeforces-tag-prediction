# Fine-tuned transformers are tuned on one fold, and ship as the bag of their fold models

A fine-tuned transformer takes 5 to 20 minutes to train on four dev folds (5 to 13 for ModernBERT-base with every weight trained, 10 to 20 for ModernBERT-large with LoRA), so a study that cross-fits every trial on the five folds, as ADR 0001 asks, would take five times as long: about 30 hours for 30 trials of ModernBERT-large with LoRA on one GPU. The two ModernBERT tagging methods, `modernbert_lora` and `modernbert_base_full` (`codeforces_tags.methods.modernbert`, first trained in the former notebooks 05 and 06, whose frozen results notebooks 06 and 07 now read), adapt the protocol of ADR 0001 to that cost:

1. **Tuning on one dev fold.** Each Optuna trial trains on dev folds 1-4 and scores fold 0. The per-tag thresholds are cross-fitted inside fold 0: its problems are cut into 5 sub-folds (whole near-duplicate groups, stratified on the focus tags), and each sub-fold is predicted with thresholds chosen on the other four. The objective is fold 0's macro F1 after the last epoch.
2. **One model per fold, one shot.** The chosen configuration trains one fold model per dev fold, on the four others, with nothing tuned again. Their out-of-fold scores, with thresholds cross-fitted across the five folds as in ADR 0001, are the method's dev rows (`predictions.csv`, `scores.csv`); the macro F1 of folds 1-4 alone is recorded beside them (`outside_tuning_fold_macro_f1`), since the configuration was chosen on fold 0.
3. **The bag is the method.** The method is a score method of the framework ([ADR 0004](0004-tagging-method-framework.md)) whose `scores` are the mean of its fold models' probabilities, and whose per-tag thresholds are chosen on every out-of-fold score. It is fixed before any test score is seen, and scores the test set once. Its weights stay in a local cache that its method folder pins by hash ([ADR 0006](0006-trained-weight-cache.md)).

Fine-tuned methods are compared with each other and with the other methods on their out-of-fold dev scores, paired on the same problems.

**Checks.** A frozen ModernBERT method is checked in three ways:

- `freeze` and `verify` re-score each dev fold with its fold model, loaded from the method folder, against the frozen out-of-fold scores, within the registry's `score_atol`. On the GPU that froze it, in the same batches, the gap is 0. Elsewhere the scores move: `score_atol` was first 5e-3, from a sample on which fp16 autocast on CUDA moved a probability by up to 2.4e-3 against another batching or float32 on the CPU. Measured on every dev problem (amended 2026-10-11), float32 on the CPU moves an out-of-fold probability by up to 8.25e-3 for `modernbert_lora` and 5.98e-3 for `modernbert_base_full`, and fp16 on the GPU one problem at a time by 6.03e-3 for `modernbert_lora`, so `score_atol` is 1e-2. `verify` re-scores on the CPU only, which keeps the GPU off a sustained load: one batch of many problems rebooted the machine once, and a loaded bag on the GPU scores one problem per forward pass.
- The migration from the notebooks reused their studies and fold models (`export`, `adopt`): no command re-runs the study. The exported files equal the notebooks', and the frozen thresholds, scores and macro F1 equal what the notebooks reported.
- `verify --retrain` trains every fold model again, each from the seed that trained it. GPU training is not reproducible bit for bit, so its pass rule is declared in the settings (`retrain_*`) before any retraining:
  - for every fold model and tag, the Pearson correlation of the retrained and frozen out-of-fold probabilities is at least 0.9;
  - their median gap is at most 0.02;
  - at most 2.5% of the dev predictions flip;
  - the interval of the paired difference of the dev macro F1 lies within ±0.03.

  Two different models of the same macro F1 fail the first three: `modernbert_lora`'s and `modernbert_base_full`'s out-of-fold scores correlate at 0.67 to 0.91 per fold model and tag (0.67 at worst), with a median gap of 0.045 and 4.9% of flips. The retrained fold 0 repeats the chosen trial's training (same configuration, seed and problems), so `verify --retrain` also prints its tuning-fold macro F1 beside the study's.

## Considered options

- **Cross-fitting every trial on the five folds** (ADR 0001 as written): a less noisy objective, at five times the cost of the study.
- **Refitting the chosen configuration on the whole dev set**, as `tuning freeze` does for the TF-IDF baseline (`fit_tag_model`): one more training, a model with no out-of-fold scores of its own, and, for a training that can collapse, no check on it. The fold models are already trained, and averaging them protects against a weak one.
- **Passing `verify --retrain` when the interval of the macro F1 difference holds 0**: two different models pass it (the difference of `modernbert_lora` and `modernbert_base_full` is -0.008 [-0.024, +0.007]).

## Consequences

- The study's objective is noisy: near-identical configurations differ by 0.026 macro F1 on fold 0, and fold 0's frozen LoRA model, trained like the chosen trial, scores 0.659 there against the trial's 0.684. The chosen trial's score overstates the configuration.
- Fold 0's model retrains the configuration chosen on fold 0, so the pooled out-of-fold score carries a little of the study's selection; the score of folds 1-4 does not.
- The bag has no dev estimate: every dev problem was seen by four of the five fold models. Its thresholds are those chosen on single models' out-of-fold scores.
- A training can collapse to uninformative scores: a fold model whose macro average precision on 800 of its own training problems stays under 0.3 is trained again with the next seed (`modernbert_lora`'s fold 1 needed seed 1). `verify --retrain` starts each fold from its frozen seed.
- **`verify --retrain` failed, twice** (`modernbert_lora`, 2026-10-08). Both runs passed the median gap (0.010 and 0.014) and the flips (1.75% and 2.15%), and failed the correlation floor (0.852 and 0.799, each at one tag of one fold model) and the macro F1 margin (−0.027 [−0.039, −0.016] and −0.019 [−0.032, −0.006]). The retrained training starts bit for bit where the frozen fold models' did: the same Trainer arguments, the same 402 initial tensors after the seed, the same batch order, and the same library versions. The differences therefore come from GPU non-determinism (fp16 atomic additions in the backward pass, compounded over about 200 optimizer steps).
  - The two retrains differ from each other as much as each differs from the frozen fold models: lowest correlation 0.807, median gap 0.014, 2.2% of flips. So no run of this training would meet the 0.9 floor: the floor was set without a measurement of this spread.
  - Their bags score 0.634 and 0.642 out of fold, close to each other (+0.007 [−0.005, +0.020]) and both under the frozen bag's 0.661. The frozen bag's dev score sits on the high side of its configuration's spread, by about 0.02, which is as large as the differences between the methods: likewise, the four trainings of fold 0 with the chosen configuration score 0.684, 0.659, 0.655 and 0.636 there.
  - The bounds stay as declared, since they are part of the recipe, so changing them would make the frozen files stale. A future rule takes its bounds from the spread of several retrains, measured first.
