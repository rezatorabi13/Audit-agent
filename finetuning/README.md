# Fine-tuned risk classifier (optional)

A 4-bit QLoRA adapter over **Qwen2.5-3B-Instruct** that classifies transactions
into the workflow's fixed category vocabulary.

## Why fine-tune here

The workflow already labels transactions with rules. This module exists to
demonstrate the case where fine-tuning genuinely beats prompting: a narrow,
high-volume classification task where **output format stability** matters as much
as accuracy.

A prompted general model that answers *"This looks like a duplicate payment
issue"* instead of `Duplicate Payment Risk` breaks every downstream parser.
Training on a fixed label set makes the format reliable in a way prompt
engineering can only approximate — and a 3B specialist is far cheaper per
transaction than a frontier model.

`LoRAClassifier` implements the same `TransactionClassifier` protocol as the
default rule-based labeller, so swapping it in changes no workflow code:

```python
from audit_agent.pipeline import build_workflow
from finetuning.classifier import LoRAClassifier

workflow = build_workflow(classifier=LoRAClassifier("data/models/risk_classifier/adapter"))
```

Outputs are snapped back onto the known vocabulary before they leave the
classifier, so an unexpected generation cannot reach downstream code.

## Requirements

- CUDA GPU with ~8GB VRAM (developed against an RTX 4070)
- Dependencies come from the project's `requirements.txt`

Qwen2.5 is Apache-2.0 and ungated, so training starts without a license
acceptance step. Swapping in a Llama checkpoint is a one-line change if you have
Hugging Face access.

If a local chat model is already resident on the GPU, unload it first — they
otherwise compete for the same VRAM:

```bash
ollama stop llama3.2:3b
```

## Training

```bash
python -m finetuning.train_qlora --output-dir data/models/risk_classifier
```

The script reports accuracy on a held-out split **before and after** training,
plus a count of off-vocabulary generations. Since LoRA's `B` matrices initialise
to zero, the "before" number is the base model's true zero-shot behaviour, which
makes the comparison meaningful rather than decorative.

Useful flags: `--epochs`, `--lora-rank`, `--examples-per-class`, `--seed`.

## Training data

`dataset.py` generates balanced, seeded examples across the four categories.
The data is synthetic — no client transactions are involved — and the seed makes
runs reproducible.
