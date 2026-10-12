# Task: tag prediction for Codeforces exercises

This file states the task the project sets out to solve. The [README](README.md) reports the methods tried and their scores.

## Goal

[Codeforces](https://codeforces.com) publishes algorithmic exercises, called problems in this project, and labels each one with tags that name the techniques it needs, such as `math`, `greedy`, `graphs` or `strings`. The task is to predict the tags of a problem from its statement and an accepted Python solution. A problem can carry several tags, or none, so this is multi-label classification.

## Data

The data is a subset of the tag classification data of [xCodeEval](https://github.com/ntunlp/xCodeEval) (Khan et al., ACL 2024), a benchmark built from Codeforces problems and the solutions submitted to them. The subset holds 4,982 problems, all distinct, stored as one `sample_<id>.json` file per problem. A file holds 21 fields:

| Fields | What they hold |
|---|---|
| `prob_desc_description`, `prob_desc_input_spec`, `prob_desc_output_spec` | The statement: the description, the input format and the output format |
| `prob_desc_sample_inputs`, `prob_desc_sample_outputs`, `prob_desc_notes` | The sample inputs and sample outputs, each the JSON text of a list of strings, and the notes that explain them |
| `prob_desc_time_limit`, `prob_desc_memory_limit`, `prob_desc_input_from`, `prob_desc_output_to`, `prob_desc_created_at` | The limits, the input and output channels, and the release date |
| `source_code`, `lang`, `lang_cluster`, `exec_outcome`, `code_uid`, `file_name`, `hidden_unit_tests` | The solution, one accepted Python program, its language version and its record in xCodeEval (`hidden_unit_tests` is empty in this subset) |
| `src_uid`, `difficulty` | The problem's id and its Codeforces rating |
| `tags` | The target: the problem's tags |

The description (`prob_desc_description`) and the solution (`source_code`) are the main inputs; any other field may be used too. The subset is not shipped in this repository: the [Data](README.md#data) section of the README explains where to put it.

## Focus tags

The dataset uses 37 distinct tags. Several are hard to predict and noisily labelled, so the project concentrates on 8 focus tags, listed here in the order the code uses, and the command line reports these 8 only:

```python
['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
```

## Goals

- **Scores per tag and in aggregate.** Every method is scored on each focus tag (precision, recall and F1), so that the tags it predicts best and worst are visible, and on the 8 together (macro and micro F1).
- **A command line.** `codeforces-tags`, in one Python module (`src/codeforces_tags/cli.py`), whose `predict` and `evaluate` commands run on a test dataset, plus the code that trains each method that learns from the data.
- **Latency.** A prediction for one problem should finish in under ten seconds.
- **A presentation deck.** Slides on the approach, the metrics and the next steps, in [`docs/slides/`](docs/slides/).
