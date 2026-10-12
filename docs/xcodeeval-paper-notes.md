# xCodeEval paper notes for the tag-prediction task

What the paper behind our dataset says that bears on [the task](../PROBLEM.md): predicting Codeforces tags for an exercise from its description and Python solution.

**Source**: Khan, Bari, Do, Wang, Parvez, Joty. *xCodeEval: An Execution-based Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval*. [arXiv:2303.03004](https://arxiv.org/abs/2303.03004) v4, 6 Nov 2023; published at ACL 2024 ([2024.acl-long.367](https://aclanthology.org/2024.acl-long.367/), pages 6766-6805).
`p.N` = PDF page N. Lines marked **(derived)** are our own arithmetic or inference, not stated in the paper.

## Key takeaways

1. **Tags are problem-level, human-written, multi-label.** They come from Codeforces and are generally updated after the contest by experienced problem solvers the Codeforces team trusts (p.18). Every solution to a problem inherits its tag set (p.3). Samples carry about 2.2–2.7 tags each **(derived)**.
2. **The bar to beat is low.** The paper's only tag-classification baseline is zero-shot `gpt-3.5-turbo-0301`. For Python it scores **29.45 macro-F1 from code only** and **33.73 from description + code**, averaged over the full ~31-tag set (p.6). This bar may be optimistic because of contamination; see §8.
3. **The description helps in every language**: +6.31 macro-F1 on average and +4.28 for Python (p.6, derived from Table 3). So feed the model both `prob_desc_description` and `source_code`.
4. **The paper's metric is macro-F1**, the unweighted mean of per-tag F1, chosen because it is independent of class size (p.29). The task statement also asks which tags are predicted best, so report per-tag F1 as well. The paper reports no per-tag scores.
5. **The tags are heavily imbalanced.** Among the focus tags, `math` has about 18× more train samples than `probabilities`, and about 22–23× more on valid/test **(derived from Fig. 8, p.24)**. Even the paper's balanced evaluation sets stay right-skewed (skew > 1.6, p.23).
6. **Splits must be problem-disjoint.** The paper holds out whole problems, not individual solutions (p.3). Our 4,982 exercises are distinct, so a random split is already disjoint. Public xCodeEval data also contains the tags of *our own* exercises, so any extra data must exclude our evaluation `src_uid`s (§7).
7. **Expect some label noise.** Tags are copied as-is from Codeforces metadata (p.5, p.39). The paper describes no relabeling and no deduplication procedure, and the data was never human-audited (p.9). The task statement (`PROBLEM.md`) also calls several tags noisily labelled.
8. **There are more useful text fields** besides the description: input/output specs, notes, sample I/O, difficulty, time/memory limits and release date (p.33).

## 1. Dataset at a glance

| Item | Value | Page |
|---|---|---|
| Source | codeforces.com only (single domain) | p.3, p.9 |
| Size | 25M document-level samples, 16.5B tokens. The introduction says "20M", which is inconsistent. | p.1, p.2 |
| Distinct problems | 7,514. Our 4,982 exercises are about 66% of them **(derived)**. | p.3 |
| Problem release dates | 19 Feb 2010 to 21 Nov 2022 | p.7, p.42 |
| Languages | 11 for classification and generation: C, C#, C++, Go, Java, Javascript, Kotlin, PHP, Python, Ruby, Rust. Retrieval adds 6 more, for 17. | p.2 |
| Tasks | 7: Tag Classification, Code Compilation, Program Synthesis, Code Translation, Automatic Program Repair, Code-Code Retrieval, NL-Code Retrieval | p.5 |
| Difficulty | Codeforces rating, 800–3500. Higher is harder. It is `None` for a few problems. | p.3, p.34, p.38 |
| Solutions | Both accepted and failing submissions are included. Execution outcomes are `PASSED`, `WRONG_ANSWER`, `COMPILATION_ERROR`, `RUNTIME_ERROR`, `TIME_LIMIT_EXCEEDED` and `MEMORY_LIMIT_EXCEEDED`. | p.4 |
| Python `lang` values | `PyPy 3`, `PyPy 3-64`, `Python 3 + libs`, `Python 2`, `PyPy 2`, `Python 3`. `lang_cluster` holds the generic language name. **(derived)** Some solutions can be Python 2, so Python 3 `ast.parse` may fail on them. Code-feature extraction needs a fallback (e.g. token or regex features). | p.30, p.34 |

## 2. Schema

### `problem_descriptions.jsonl`: one record per problem (p.33)

When these columns are copied into task records, most get a `prob_desc_` prefix. `tags`, `src_uid` and `difficulty` stay unprefixed (pp.35–37).

| Field | Name in task records | Meaning |
|---|---|---|
| `description` | `prob_desc_description` | Problem statement, with math in LaTeX |
| `input_spec` | `prob_desc_input_spec` | Input format and order: ranges, types, sizes |
| `output_spec` | `prob_desc_output_spec` | Output format. Checked by exact string match or a float tolerance. |
| `sample_inputs` / `sample_outputs` | `prob_desc_sample_inputs` / `prob_desc_sample_outputs` | Example I/O |
| `notes` | `prob_desc_notes` | Explanation of the samples |
| `time_limit` / `memory_limit` | `prob_desc_time_limit` / `prob_desc_memory_limit` | Resource limits |
| `input_from` / `output_to` | `prob_desc_input_from` / `prob_desc_output_to` | I/O channels |
| `created_at` | `prob_desc_created_at` | Unix timestamp of the problem's release |
| `tags` | `tags` | Problem categories, the **target** |
| `difficulty` | `difficulty` | Expert-annotated difficulty rating |
| `src_uid` | `src_uid` | Unique problem id, a 32-hex string (e.g. `db884d679d9cfb1dc4bc511f83beedda`) |

### Task records that matter (pp.34–37)

- **`tag_classification`** records have exactly 7 fields: `lang`, `source_code`, `tags` (the label), `lang_cluster`, `code_uid`, `src_uid`, `difficulty` (p.34).
  - They contain **no `prob_desc_*` columns**. The paper says `src_uid` "is important for the training of the model" because it joins a sample to the problem's description (p.34).
  - `code_uid` "is not important for model training" (p.34).
  - There is no `exec_outcome` field. The `src_uid` description says it points to the problem "that the code successfully solved" (p.34), but that sentence is boilerplate reused in other tasks.
  - **(derived)** Tag-classification train sizes nearly match the all-`PASSED` Program Synthesis train (Python: 678,576 vs 681,780; Table 2, p.5; p.24). `PROBLEM.md` also describes the solutions as accepted. So assume the solutions are accepted, and check `exec_outcome` if our subset has it.
- **`program_synthesis`** and **`code_translation`** records carry the 11 `prob_desc_*` columns, plus `exec_outcome`, `file_name` and `hidden_unit_tests`, but they have **no `tags`** (pp.35–37).
- **`apr`** records carry `tags` **and** all 11 `prob_desc_*` columns. Their code is in `bug_source_code` / `fix_source_code`, with outcomes in `bug_exec_outcome` / `fix_exec_outcome`. There is no plain `source_code` field (pp.34–35).

**(derived)** Our subset has `prob_desc_description`, `source_code` and `tags` together, so it was probably built by joining `tag_classification` with `problem_descriptions.jsonl` on `src_uid`. Check which other `prob_desc_*`, `difficulty` and `exec_outcome` columns survived before choosing features.

## 3. Tags

### Taxonomy: the 31 tags shown in Figure 8 (p.24)

`2-sat`, `binary search`, `bitmasks`, `brute force`, `combinatorics`, `constructive algorithms`, `data structures`, `dfs and similar`, `divide and conquer`, `dp`, `dsu`, `expression parsing`, `fft`, `flows`, **`games`**, **`geometry`**, `graph matchings`, **`graphs`**, `greedy`, `implementation`, `interactive`, **`math`**, `matrices`, `meet-in-the-middle`, **`number theory`**, **`probabilities`**, `shortest paths`, `sortings`, **`strings`**, **`trees`**, `two pointers`. The 8 focus tags of this project are in bold.

- Tags describe the algorithmic techniques needed to solve a problem. A problem often has several, because it can be solved in more than one way or needs a combination of techniques (p.3, p.24).
- Example from Figure 1: the "watermelon" problem has tags `brute force, math` and difficulty 800 (p.2).
- **(derived)** A problem's tags can cover several alternative solutions, and one submission usually implements only one of them. So a code-only model has a built-in ceiling: some gold tags simply cannot be seen in a given solution. That is a strong argument for combining the description with the code.

### Distribution of tag-classification samples (Figure 8, p.24)

The counts below are **per code sample**, summed over 11 languages. They are read off log-scale bars, so they are approximate. Shares are counts divided by the Table 2 split totals **(derived)**.

| Focus tag | Train | Train share | Valid | Test | Test share |
|---|---|---|---|---|---|
| math | ~1.53M | 27.7% | ~7.1k | ~28.3k | 37.9% |
| graphs | ~544k | 9.9% | ~600 | ~2.6k | 3.5% |
| strings | ~498k | 9.1% | ~2.2k | ~8.9k | 11.9% |
| number theory | ~381k | 6.9% | ~1.7k | ~7.0k | 9.4% |
| trees | ~363k | 6.6% | ~290 | ~1.7k | 2.3% |
| geometry | ~164k | 3.0% | ~530 | ~2.1k | 2.8% |
| games | ~99k | 1.8% | ~360 | ~1.7k | 2.3% |
| probabilities | ~84k | 1.5% | ~300 | ~1.3k | 1.7% |

- **Largest tags overall**: `implementation` and `greedy` (~1.86M each), then `math` (~1.53M), then `data structures`, `dp` and `constructive algorithms` (~1M each), then `brute force` (~0.9M).
- **Smallest tags**: `expression parsing`, `meet-in-the-middle`, `2-sat` (~15–17k each).
- `2-sat` has no validation or test samples at all (p.24).
- Label cardinality is about 2.67 tags per sample in train and 2.2–2.3 in validation/test **(derived)**.
- **Caveat (derived):** these counts are per submission. Popular problems have thousands of solutions, so the shares are weighted toward them. Prevalence per problem in our 4,982-exercise subset will differ, so measure it on our data.
- **Train/eval shift (derived):** tag shares differ between splits. `trees` goes from 6.6% (train) to 1.5% (valid), and `implementation` from 34% to 53%. Two likely causes:
  - the held-out set is a different population of problems: those with complete unit tests (p.20);
  - circulation selection caps samples per problem and per tag (p.4, p.22).

  Either way, thresholds tuned on one distribution may not transfer to another.

## 4. The task as the paper defines it (p.5, p.22)

- **TC-Code2Tag**: given code `C` in any language, predict its tag set `T`.
- **TC-DesCode2Tag**: given code and the natural-language problem description, predict `T`.
- The gap between the two settings measures how much the description helps. The paper also defines language-specific versions of both.
- It is the only xCodeEval task that ignores whether the code executes. It was included because tags already drive the sampling algorithm (p.5, p.18).
- Our task corresponds to **DesCode2Tag restricted to Python**, scored on 8 tags instead of ~31.

## 5. Baseline results (Table 3, p.6)

The model is `gpt-3.5-turbo-0301`, used zero-shot with direct prompting (no chain-of-thought), temperature 0.325, n=1, on the **validation** split (p.6, p.29). Scores are macro-F1 over all tags.

| Setting | C | C# | C++ | Go | Java | JS | Kotlin | PHP | **Python** | Ruby | Rust | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Code2Tag | 32.37 | 26.91 | 40.58 | 23.06 | 31.58 | 19.35 | 33.95 | 15.25 | **29.45** | 23.64 | 24.04 | **27.29** |
| DesCode2Tag | 36.05 | 33.18 | 47.10 | 31.50 | 38.26 | 27.81 | 39.61 | 19.36 | **33.73** | 30.61 | 32.35 | **33.60** |
| Gain (derived) | +3.68 | +6.27 | +6.52 | +8.44 | +6.68 | +8.46 | +5.66 | +4.11 | **+4.28** | +6.97 | +8.31 | **+6.31** |

- The paper reports **no** fine-tuned or classical classifier for tags, **no** per-tag scores, and **no** micro-F1 (p.6, p.29).
- The Code2Tag/DesCode2Tag prompts are never shown. Where they should appear, p.29 has a leftover author TODO.
- The temperature was not tuned for this task. It was carried over from the program-synthesis sweep (p.8, p.29). The baseline is a single stochastic run with no variance reported.
- **(derived)** These numbers are not directly comparable to ours: they cover a different tag set (~31 tags), multiple solutions per problem, and the paper's balanced validation split. Treat them as context, not a leaderboard.

## 6. Metric (Appendix F, p.29)

- Per tag: `F1_T = 2·TP_T / (2·TP_T + FP_T + FN_T)` (formula cited to Taha & Hanbury, 2015). Macro-F1 is the unweighted mean of `F1_T` over the tag set (cited to Opitz & Burst, 2019).
- The paper justifies macro averaging as independent of class population size.
- The paper says nothing about thresholds, or about a tag with `TP = FP = FN = 0`.
- **(derived)** With `sklearn.metrics.f1_score(average="macro")`, set `zero_division` explicitly. Report per-tag precision, recall, F1 and support alongside the macro and micro averages.
- **(derived)** A 20% hold-out of our data is about 1,000 problems. If per-problem prevalence is close to the per-sample shares above, rare focus tags (`probabilities`, `games`, `geometry`) would have only about 15–30 positives, so their per-tag F1 will be noisy. Use stratified k-fold CV or bootstrap confidence intervals over problems, and check the actual prevalence first.
- For comparison, the other xCodeEval tasks use accuracy (compilation), pass@k (generation) and Acc@k (retrieval).

## 7. Splits, data selection and extra data

- **Held-out problems**: 1,354 of the 7,514 problems were set aside for validation and test, so evaluation problems are never seen in training (p.3). Problems with complete unit tests went to the held-out set; the rest went to train (p.20).
- **Algorithm 1** (p.21) splits the held-out set into validation and test:
  - Shuffle the held-out set over many seeds and cut it at ratio γ.
  - Reject any split where validation and test have different tag sets.
  - For each tag T, compute `γ_T = #valid(T) / #test(T)`.
  - Keep the split whose geometric mean of the `γ_T` is closest to γ.
  - For tag classification, γ = 0.2, i.e. 1:5 (p.22).
- **Circulation-based selection** (p.4, p.22) then subsamples the selected data:
  - Build a flow network source → problems → tags → sink, with lower and upper bounds on samples per problem (`m_p`, `x_p`) and per tag (`m_t`, `x_t`).
  - A feasible flow gives the number of samples to take per problem and per tag.
  - The targets are M = 2,000 validation and 10,000 test samples per language. The tuple `(m_p, m_t, x_p, x_t) = (1, 1, 1000, 1000)` is valid, and Δ = 1000.
- **Resulting sizes** (Tables 2 and 8, pp.5, 23) are code samples, not problems:

  | | Train | Valid | Test |
  |---|---|---|---|
  | All 11 languages | 5,494,008 | 18,696 | 74,733 |
  | Python | 678,576 | 1,969 | 8,195 |

  C++ dominates training with 3.71M samples. Java (704k) and Python (679k) come next.
- **Effect on skew** (Table 7, p.23, Python tag classification): circulation selection is compared with random selection of the same size (p.22). It gives lower skew: 2.365 vs 2.731 on validation, 2.173 vs 2.689 on test. Standard deviation is also lower: 240.2 vs 265.2 (validation) and 992.9 vs 1125.1 (test). The distribution still stays right-skewed.
- **(derived) What we can reuse:** use a multi-label stratified split (e.g. iterative stratification), or copy the paper's trick of searching seeds for the per-tag ratio closest to the target. Either way, rare tags (`probabilities`, `games`, `geometry`) stay present in every fold. Also check that every focus tag appears in train, validation and test.
- **(derived) Extra data and leakage:** `problem_descriptions.jsonl` stores `tags` for every xCodeEval problem (p.33).
  - That gives about 7,514 − 4,982 ≈ 2,500 labelled problems outside our subset. These are most valuable for rare tags.
  - The tag-classification train split also has 678,576 Python solutions (p.5), on the order of 100 per problem. These could be used as code-side augmentation.
  - The same file holds the tags of our own exercises. Any external xCodeEval data must exclude every `src_uid` in our validation and test folds, or test labels leak.

## 8. Data quality and leakage

- **How labels were made**: they are human-written (p.18) and "come as a metadata from the sources" (p.39). No software was used for labeling (p.40), and the raw data was not kept (p.39). The paper describes no relabeling.
- **Cleaning**: automated tools removed about 2M samples with sensitive content (p.9): emails, large infographics, toxic keywords. The data was also "de-anonymized", which presumably means anonymized (p.39). No other cleaning step and no deduplication procedure is described. The paper does state that code samples are unique for synthesis and translation (p.24).
- **The authors' own claim**: to their knowledge there are "not errors, sources of noise or redundencies" [sic] (p.38). Treat this with caution, since the task statement calls several tags noisily labelled.
- **Missing or invalid values**:
  - `difficulty` is `None` for a few problems (p.38), even though the nutrition label claims 0% missing (p.42).
  - NL-Code retrieval keeps only problems whose description "is valid" (p.26), which implies some descriptions are not. **(derived)** Check for empty or very short `prob_desc_description`, and fall back to code-only features when it happens.
- **Limitations the paper lists** (p.9): a single source; large imbalance between languages; whole-file code that is often non-modular and has no docstrings.
- **Contamination**:
  - ChatGPT's C++ *program-synthesis* performance drops sharply on problems released after its Sep-2021 knowledge cutoff (Fig. 4, p.7). The paper reads this as potential contamination (p.7, p.41).
  - The paper gives no cutoff analysis for tag classification.
  - The held-out validation/test problems are mostly older: dense from 2011 to about 2020, rare afterwards. Training problems dominate 2020–2022 (Fig. 7, p.21). This is a side effect of choosing problems with complete unit tests, not a temporal split.
  - **(derived)** So the ChatGPT tag scores were measured mostly on problems released before its cutoff, and may be inflated by memorization of problems *and their tags*. Any LLM-based approach of ours has the same risk. Slice evaluation by `prob_desc_created_at` to check.

## 9. Modeling hints the paper offers

- **Input text template** (from NL-Code retrieval, p.30): `Description: {description} Input specification: {input_spec} Output specification: {output_spec}`. This is a ready-made way to join the problem fields.
- **Encoder**: the only fine-tuned encoder in the paper is StarEncoder, used in a DPR-style bi-encoder (max length 1024, batch 48, 37 epochs; p.30).
  - Fine-tuned, it reaches Acc@100 of 84.57 for Python NL-Code (Table 4, p.8) and 90.1 for Python→Python Code-Code (Fig. 12, p.28).
  - **(derived)** So same-problem text and code land close together in its embedding space. That supports a kNN or linear-probe tag classifier on these embeddings. Note that retrieval measures problem identity, not tag similarity.
- **Sequence length**: no length statistics are given for tag classification. For APR inputs (buggy code + description, bloom tokenizer), the median is about 510 tokens, the 95th percentile about 1,565 and the 99th about 3,420 (p.26). **(derived)** Expect truncation trade-offs with 512-token encoders.
- **Difficulty**: ChatGPT's program-synthesis success falls sharply as difficulty rises (p.8). Difficulty is available as a feature, but the paper does not link it to tags.

## 10. Access and licensing

- **Data**: <https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval>, in jsonl/json format, with an arrow loader for `datasets`.
  - The root holds `problem_descriptions.jsonl` and `unittest_db.json`, both keyed by `src_uid` (p.33, p.42).
  - The per-task data file names are listed in the Hugging Face loader script, `xCodeEval.py` (p.42).
- **Code**: <https://github.com/ntunlp/xCodeEval>. The execution engine is <https://github.com/ntunlp/ExecEval>.
- **License**: CC BY-NC 4.0, released March 2023 (p.42). See the Licensing section of the GitHub repo (p.40).

## 11. Inconsistencies in the paper

- **Dataset size**: 25M in the abstract, 20M in the introduction. Languages: 11 vs 17, where 17 applies only to retrieval (p.1–2).
- **γ = 0.2**: it is used both as the slice fraction (≈1:4) and as the target ratio (1:5). The final tag-classification totals are 18,696 : 74,733 ≈ 1:4 (p.21–23).
- **Tag coverage**: the text says validation/test share the training tag set (p.20). The pseudo-code only checks that validation and test match each other, and `2-sat` has no validation/test samples (p.24).
- **Table 7**: it reports tag-classification skew for only 5 of the 11 languages (p.23).
- **Temperature**: 0.32 vs 0.325, and 0.325 is not on the `np.linspace(0, 2, 20)` grid. The setting is spelled both "DesCode2Tag" and "DescCode2Tag" (p.8, p.22, p.29).

## 12. What the paper does not provide

None of the following is in the paper, so all of it is ours to produce:

- per-tag scores, micro-F1, or any variance or confidence intervals;
- a threshold strategy, or any trained tag classifier;
- the prompts used for the baseline;
- sequence-length statistics for tag classification;
- tag co-occurrence statistics;
- a described deduplication procedure;
- any analysis of this project's 8-tag subset.
