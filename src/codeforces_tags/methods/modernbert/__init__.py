"""Two fine-tuned ModernBERT tagging methods, each the bag of its five fold models (docs/adr/0005).

``modernbert_lora`` trains LoRA updates on ModernBERT-large; ``modernbert_base_full`` trains every
weight of ModernBERT-base. Both tune on one dev fold, train one model per dev fold with the chosen
configuration, and predict with the mean of their fold models' probabilities. Their trained weights
are too large for a method folder: they live in the trained cache, which the folder's
``trained.json`` names with their hashes (docs/adr/0006).

The prediction path (``settings``, ``inputs``, ``weights``, ``modeling``, ``training``,
``method``) never imports mlflow or optuna; ``tuning`` and ``__main__`` do.
"""
