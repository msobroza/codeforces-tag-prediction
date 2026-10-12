# Trained weights too large for a method folder stay in a local cache, pinned by their hashes

A ModernBERT method is the bag of five fold models ([ADR 0005](0005-fine-tuned-model-protocol.md)): five LoRA adapters of 59 MiB for `modernbert_lora`, five whole ModernBERT-base models of 571 MiB for `modernbert_base_full`. A method folder holds at most 50 MiB per file and 100 MiB in all (`check_method_folder`, [ADR 0002](0002-method-folders-and-mlflow-wrapper.md)), and GitHub refuses files over 100 MB, so these weights are not committed. They stay in a gitignored cache, as the pretrained snapshot does ([ADR 0004](0004-tagging-method-framework.md)):

- the trained cache is `experiments/trained/` (`config.TRAINED_CACHE_DIR`), or `$CODEFORCES_TRAINED_DIR`, read when a method loads; the method's `train` or `adopt` command fills it, one folder per fold model;
- the method folder commits the cache's weight list, `trained.json`: each fold model's file, relative to the cache, with its bytes and sha256, beside `pretrained.json`, the weight list of the pretrained snapshot;
- loading checks every hash before building the bag. A missing file raises an error that says how to fill the cache, and a changed file raises another.

A pinned snapshot is downloaded again byte for byte (`fetch`), but fold models trained on a GPU are not reproducible bit for bit. A frozen method's fold models are therefore restored by copying its trained cache from the machine that froze it, with `trained.json` as the list of what to copy. On that machine, `adopt` restores them from the notebooks' runs. Training again makes other weights: a new method, frozen under a new name. `train` refuses the cache folder of a frozen method, and `verify --retrain` trains into the cache's `retrain/` folder.

## Considered options

- **Committing the weights in a smaller form**: in fp16, the five LoRA adapters still take 150 MiB, over the folder cap, and a whole ModernBERT-base model takes 285 MiB.
- **Git LFS, or a release asset**: the weights would travel with the repository, but they would need a storage quota and an external service at load time, for 3.2 GB of models trained on the challenge's data.
- **The Hugging Face Hub, pinned by revision like the pretrained snapshot**: it would publish models trained on the challenge's data.
- **MLflow artifacts**: the MLflow store is local and gitignored too, so it moves the weights without making them portable, and a logged model would hold up to 2.9 GB.

## Consequences

- A fresh clone cannot load the ModernBERT methods until their trained cache is copied in. Their frozen files (predictions, scores, run record, latencies, tuning results) are committed and read without the weights, so the notebooks and the comparisons of methods work anywhere.
- A logged MLflow model of a ModernBERT method needs both caches on the machine that loads it.
- `verify` re-scores with the cached weights. `verify --retrain` compares retrained fold models with the frozen ones on bounds declared beforehand ([ADR 0005](0005-fine-tuned-model-protocol.md)), never by hash.
