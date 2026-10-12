# Codeforces tag prediction

Predicting which of eight algorithmic categories a Codeforces exercise belongs to, from its statement and an accepted solution. This file fixes the words used in the code, notebooks and slides.

## Problems and their parts

**Problem**:
One Codeforces exercise, a single entry of the dataset, identified by its sample id.
_Avoid_: example, exercise, sample (it collides with sample input/output)

**Challenge dataset**:
The 4,982 problems that `PROBLEM.md` describes, a subset of xCodeEval's tag-classification data, pinned by `config.DATASET_SHA256`; the frozen split divides it into the dev and test parts.
_Avoid_: the dataset (when a copy with other problems may be meant)

**Statement**:
The description, input specification and output specification of a problem, joined; the problem text a model reads.
_Avoid_: description (only the first of the three parts), prompt

**Solution**:
The accepted Python program stored with a problem.
_Avoid_: submission, code (when the program as a whole is meant)

**Sample input / sample output**:
The worked examples printed in a problem's statement page, with their expected answers.
_Avoid_: test case, unit test (those are the hidden tests)

**Notes**:
The explanation of the samples printed under a problem's examples; part of the problem page, but not of the statement.
_Avoid_: remarks, explanation

**Release era**:
Whether a problem came out up to 2017 or from 2018 on, the year statements switched from flattened Unicode math to `$$$` LaTeX; used to check what the model learns from release-time artefacts rather than content.
_Avoid_: period, epoch, LaTeX era

**Self-identifying solution**:
A solution whose code names or quotes its own problem (a Codeforces URL, a contest id, or copied statement text); its problem forms the contamination-sensitivity slice.
_Avoid_: leaked problem

## Model inputs

**Feature block**:
The features derived from one input of a problem: the statement block (statement text, with notes optionally), the code block (the solution's tokens) and the cue block (sample I/O cues), and, for the embedding methods, the embedding blocks (the statement and solution embeddings).
_Avoid_: feature group, view, modality

**Statement input / solution input**:
The two texts of a problem that the encoder reads: the statement with its notes, and the solution's source code.
_Avoid_: chunk, field, document

**Encoder input**:
An input with its instruction prepended: the exact string the tokenizer receives.
_Avoid_: model input (the heading "Model inputs" already groups the feature blocks), prompt, query

**Embedding block**:
The feature block built from one input's embedding (`statement_embedding`, `solution_embedding`), beside the TF-IDF feature blocks `statement` and `code`.
_Avoid_: embedding features, vector block

**Sample I/O cue**:
A yes/no fact read from a problem's sample input or sample output, such as "the sample input is a tree's edge list" or "the sample output is a decimal number".
_Avoid_: sample-output cue, output feature, hand-crafted feature

**Tag model**:
The fitted feature blocks and linear models together with the per-tag thresholds: what turns a problem into its predicted focus tags.
_Avoid_: tagger, classifier, predictor

## Tagging methods

**Tagging method**:
Anything that turns raw problems into 0/1 tags through `predict`: the tag model, a prompted method, or any other registered method.
_Avoid_: predictor, tagger, model (unqualified)

**Method kind**:
How a tagging method gets honest dev predictions: a score method (cross-fitted thresholds), a fitted 0/1 method (`cross_predict`), a zero-shot method (nothing learned from dev), or another kind.
_Avoid_: method type, family

**Method folder**:
The saved form of a tagging method: `method.json` plus the method's own data files, loadable without MLflow.
_Avoid_: model artifact, checkpoint

**Run record**:
The JSON file that pins what a frozen result was computed from: code, data, environment and settings fingerprints.
_Avoid_: manifest, metadata

**Prompted method**:
A tagging method that asks an LLM through the toolkit, with a fixed prompt, request settings and output format, e.g. `descode2tag-json`.
_Avoid_: LLM model, variant

**Prompt file**:
One version of an API method's prompt, as one YAML file of its package, `templates/<name>.v<N>.yaml`: its name, its version and its words (a Jinja2 template, a schema, tag definitions), never edited once a request was sent, so a new wording is a new file.
_Avoid_: prompt config, prompt template (for the whole file), prompt registry entry

**Output format**:
How a prompted method's answer is returned: `text` (free text read by a fixed parser) or `json` (a strict JSON schema).
_Avoid_: response mode

**Answer**:
One frozen API response for one problem and method, keyed by its request hash.
_Avoid_: completion, generation

**Zero-shot setting**:
One of the paper's two LLM tasks, Code2Tag (the solution alone) or DesCode2Tag (the statement, notes and solution).
_Avoid_: LLM baseline, prompt variant

**Zero-shot method**:
A prompted method that learns nothing from dev: a zero-shot setting with an output format, e.g. `descode2tag-json`.
_Avoid_: LLM model, variant

**Fold model**:
A fine-tuned model trained on four dev folds, whose scores of the fifth are its out-of-fold scores.
_Avoid_: fold adapter, split model, CV model

**Bag**:
A fine-tuned method's five fold models, whose mean probability is the method's score.
_Avoid_: ensemble (unqualified), committee

**Weight list**:
A method folder's `pretrained.json` or `trained.json`: each weight file the method reads from a local cache, with its sha256.
_Avoid_: manifest, checkpoint list

## Labels

**Tag**:
A Codeforces category attached to a problem by the site.
_Avoid_: label (when a category name is meant), class

**Focus tag**:
One of the eight tags the task statement (`PROBLEM.md`) asks to predict: math, graphs, strings, number theory, trees, geometry, games, probabilities.
_Avoid_: target, main tag

**Tag definition**:
The extended meaning of a focus tag that a method sends with its question: what Codeforces means by the tag, what it includes, and what it excludes, against its neighbouring tags.
_Avoid_: label description, tag prompt

**No-focus problem**:
A problem none of whose tags is a focus tag, including the problems with no tag at all.
_Avoid_: off-focus problem, zero-focus problem, negative problem, unlabelled problem

**Focus-tagged problem**:
A problem with at least one focus tag.
_Avoid_: positive problem, labelled problem

**Near-duplicate group**:
The problems that are one problem in different versions (easy/hard, or reused); a problem with no duplicate forms a group on its own.
_Avoid_: cluster, duplicate pair

## Evaluation

**Test set**:
The frozen fifth of the problems (`part == "test"` in the frozen split), set aside before any modelling and scored once by the chosen model; a fixed reference model that nothing is chosen from may report its single test score beside it (ADR 0001).
_Avoid_: holdout, validation set

**Dev set**:
The other four fifths: the only problems that tuning, model choice and training see.
_Avoid_: train set, training data

**Fold**:
One of the five frozen parts of the dev set; a near-duplicate group never spans two folds, nor the dev and test sets.
_Avoid_: split, partition

**Out-of-fold score**:
The score a dev problem gets from a model trained on the four folds that do not contain it.
_Avoid_: validation prediction, CV prediction

**Per-tag threshold**:
The score above which a focus tag is predicted; each focus tag has its own.
_Avoid_: cutoff, decision boundary

**Cross-fitted threshold**:
A per-tag threshold applied to one fold after being chosen on the out-of-fold scores of the other four, so the score it yields is not fitted to itself.
_Avoid_: tuned threshold, nested threshold

**Threshold rule**:
How a score method turns a tag's scores into 0/1: the cross-fitted F1-maximising threshold for every method but the calibrated Jev twins, which map the scores to calibrated probabilities with a Platt map, cut at half the best F1 (Lipton's rule) and snap the threshold to Jev's grid.
_Avoid_: calibrated threshold, decision rule, cutoff rule

**Calibrated probability**:
A score mapped to a probability that holds on average: of the problems given probability p for a tag, about a share p have it. The map is fitted on the out-of-fold scores of the other folds.
_Avoid_: calibrated score, calibrated threshold

**Macro F1**:
The unweighted mean of the eight per-tag F1 scores; the primary metric.
_Avoid_: F1 (unqualified), average F1

**Micro F1**:
F1 over the true positives, false positives and false negatives pooled across the eight focus tags.
_Avoid_: global F1, overall F1

**All-tag macro F1**:
The macro F1 over every tag of the 37 with at least one positive in the part scored (37 on dev, 36 on test); the headline number beside the paper's. Two variants are reported beside it: over all 37 with an absent tag's F1 counted as 0, and over the paper's 31 tags. Used only to compare with the paper.
_Avoid_: macro F1 (unqualified) for it

**Group bootstrap**:
Resampling near-duplicate groups with replacement to get confidence intervals, so the two versions of a problem are drawn together.
_Avoid_: bootstrap (unqualified), row bootstrap

**Per-tag confusion matrix**:
For one focus tag, the counts of true positives, false positives, false negatives and true negatives.
_Avoid_: confusion matrix (unqualified)

**Tag confusion matrix**:
Rows are the true focus tags plus "no focus tag", columns the predicted focus tags plus "nothing predicted"; a cell is the share of the row's problems that received the column's prediction.
_Avoid_: confusion matrix (unqualified), co-occurrence matrix

**Evaluation dataset**:
MLflow's copy of the problems of one part, with their inputs and expected tags.
_Avoid_: eval set

**Evaluation run**:
One MLflow run that scores one method's frozen predictions on one part, one trace per problem.
_Avoid_: eval job

## Searching a method's words

**Search**:
GEPA's run over the words of a TypeSafe Jev method: from a frozen method's words it proposes candidate words on the learning folds, scores them on the selection fold and chooses one by a rule fixed before it runs; its requests are records of the search, not a method's answers, and the chosen words become a new method (ADR 0010).
_Avoid_: optimisation, prompt tuning, training, study (the methods compared before a test request, or an Optuna study)

**Learning folds**:
The dev folds that a search draws its minibatches from and shows the reflection model (folds 1-3); GEPA calls them its trainset.
_Avoid_: train set, training data

**Selection fold**:
The dev fold on which a search scores every accepted candidate words in full and picks the chosen ones (fold 4); GEPA calls it its valset.
_Avoid_: validation set, dev set (all five folds)

**Held-out fold**:
The dev fold that a search never reads (fold 0), on which the searched method is compared, out of sample for its words, with the method it started from.
_Avoid_: test fold, holdout, test set (the frozen fifth of the problems)

**Candidate words**:
One set of the words a Jev method sends, the convention and the eight tag definitions (its nine pieces) with the fixed question, that a search scores; the starting method's words are candidate 0, and GEPA calls each set a candidate.
_Avoid_: prompt, candidate prompt, program

**Reflection model**:
The LLM that reads a minibatch's worst errors on one piece of the candidate words and proposes a rewrite of that piece (`gpt-6-luna` at high reasoning effort).
_Avoid_: optimizer, teacher, judge (an LLM of the near-duplicate audit)

**Balanced Brier score**:
The mean over the eight focus tags of half the mean squared error of a method's probabilities on the tag's positives plus half that on its negatives, so that a rare tag's positives count as much as its negatives; lower is better, and a search maximises one minus its per-problem form.
_Avoid_: Brier score (unqualified), loss

## Serving

**Model server**:
The HTTP server that serves one packaged tagging method: MLflow's scoring server (FastAPI on uvicorn), answering `/ping` and `/invocations`; in the container it sits behind nginx on port 8080.
_Avoid_: endpoint (SageMaker's deployment), inference server, API (unqualified)
