# Fine-tuning Qwen2.5-3B with QLoRA as a risk classifier

A 4-bit QLoRA adapter over **Qwen2.5-3B-Instruct** that classifies transactions
into the workflow's fixed category vocabulary: `Normal`, `Duplicate Payment Risk`,
`Related Party`, `Unusual Timing or Amount`.

No trained adapter is shipped with the repository. You produce it locally with
one command, and the workflow picks it up from a fixed path.

## Why fine-tune here

The workflow can already label transactions with rules. This module shows the
case where fine-tuning genuinely beats prompting: a narrow, high-volume
classification task where **output format stability** matters as much as
accuracy.

A prompted general model that answers *"This looks like a duplicate payment
issue"* instead of `Duplicate Payment Risk` breaks every downstream parser.
Training on a fixed label set makes the format reliable in a way prompt
engineering can only approximate, and a 3B specialist is far cheaper per
transaction than a frontier model.

## Requirements

- NVIDIA GPU with ~8 GB VRAM (developed on an RTX 4070)
- `pip install -r requirements.txt` (the single requirements file includes torch,
  transformers, peft, bitsandbytes, accelerate and datasets, and pulls a CUDA
  build of PyTorch)

Check that torch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, reinstall a CUDA build:
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.

Qwen2.5 is Apache-2.0 and ungated on Hugging Face, so the first run downloads it
without any license-acceptance step.

## Train

From the repository root:

```bash
python -m finetuning.train_qlora
```

What it does:

1. Unloads the Ollama model from the GPU (skip with `--keep-ollama`).
2. Generates a balanced, seeded synthetic dataset (`dataset.py`) and splits it
   80/20 into train and evaluation sets.
3. Loads Qwen2.5-3B-Instruct in 4-bit NF4 and attaches LoRA (rank 8, alpha 16)
   to `q_proj`, `k_proj`, `v_proj`, `o_proj`.
4. Measures accuracy **before** training. LoRA's `B` matrices start at zero, so
   this is the base model's true zero-shot behaviour.
5. Trains with loss on the label tokens only (prompt tokens are masked).
6. Measures accuracy and the rate of exact-label outputs **after** training.
7. Saves the result:

| Output | Default path |
|---|---|
| LoRA adapter + tokenizer | `data/models/qwen_risk_classifier/adapter/` |
| Metrics (before/after) | `data/models/qwen_risk_classifier/metrics.json` |

`data/models/` is git-ignored, so trained weights never end up in a commit.

Flags: `--epochs`, `--lora-rank`, `--examples-per-class`, `--learning-rate`,
`--batch-size`, `--seed`, `--output-dir`, `--base-model`, `--keep-ollama`.
If you change `--output-dir`, set `AUDIT_DATA_DIR` accordingly or pass the
adapter path to `LoRAClassifier` yourself.

## Use it in the workflow

`LoRAClassifier` implements the same `TransactionClassifier` protocol as the
rule-based default, so switching changes no workflow code:

```bash
python examples/03_vendor_review_qlora.py
# or for any entry point
AUDIT_CLASSIFIER=qlora python examples/02_vendor_review.py
```

or in code:

```python
from finetuning.classifier import LoRAClassifier
from audit_agent.pipeline import build_workflow

workflow = build_workflow(
    classifier=LoRAClassifier("data/models/qwen_risk_classifier/adapter")
)
```

If the adapter is missing, `AdapterNotFoundError` is raised immediately with the
training command. Outputs are snapped back onto the known vocabulary before they
leave the classifier, so an unexpected generation can't reach downstream code.

## Train/serve consistency

A classifier only sees the text it is given. A single ledger row can't reveal a
related party (that lives in the entity graph) or a duplicate (that needs the
rest of the ledger). `audit_agent.workflow.classifier.describe_transaction`
therefore writes those facts into the description, and **the same function
renders the training examples**. The adapter is trained on exactly the format it
is served.
