"""Fine-tune Qwen2.5-3B-Instruct with 4-bit QLoRA as a transaction risk classifier.

Run from the repository root (needs a CUDA GPU with ~8 GB VRAM):

    python -m finetuning.train_qlora

By default the adapter is written to ``data/models/qwen_risk_classifier/adapter``
(``Settings.qlora_adapter_dir``), which is exactly where the workflow looks for
it when run with ``AUDIT_CLASSIFIER=qlora``. Nothing trained is committed to the
repository; every user produces their own adapter with this script.

The script reports accuracy on a held-out split before and after training, so
the adapter's contribution is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

from audit_agent.config import get_settings
from audit_agent.logging_config import configure_logging

from finetuning.dataset import (
    SYSTEM_PROMPT,
    LabelledTransaction,
    generate_examples,
    train_eval_split,
)

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 256
MAX_NEW_TOKENS = 12
PROMPT_LABEL_MASK = -100


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train the QLoRA transaction classifier (Qwen2.5-3B-Instruct)."
    )
    parser.add_argument("--base-model", default=settings.qlora_base_model)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.qlora_output_dir,
        help="Run directory; the adapter is saved to <output-dir>/adapter "
        f"(default: {settings.qlora_output_dir})",
    )
    parser.add_argument("--examples-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-ollama",
        action="store_true",
        help="Do not ask Ollama to unload its model before training.",
    )
    return parser.parse_args()


def unload_ollama_model(model: str) -> None:
    """Free VRAM held by a resident Ollama model; training needs most of 8 GB."""
    if shutil.which("ollama") is None:
        return
    result = subprocess.run(["ollama", "stop", model], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Asked Ollama to unload %s from the GPU", model)


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            f"QLoRA dependencies are missing ({exc.name}). Install them with:\n"
            "    pip install -r requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA training needs a CUDA GPU, but torch.cuda.is_available() is False.\n"
            "If you have an NVIDIA GPU, you likely have the CPU-only torch build; see "
            "'Fine-tune the classifier' in README.md."
        )

    from audit_agent.workflow.classifier import CATEGORIES

    if not args.keep_ollama:
        unload_ollama_model(get_settings().ollama_model)

    examples = generate_examples(per_class=args.examples_per_class, seed=args.seed)
    train_examples, eval_examples = train_eval_split(examples)
    logger.info("%d training / %d evaluation examples", len(train_examples), len(eval_examples))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    def format_prompt(description: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transaction: {description}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict(description: str) -> str:
        inputs = tokenizer(format_prompt(description), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def evaluate(dataset: list[LabelledTransaction]) -> dict[str, float]:
        predictions = [predict(item.description) for item in dataset]
        correct = sum(
            item.label.lower() in pred.lower() for item, pred in zip(dataset, predictions)
        )
        exact = sum(pred in CATEGORIES for pred in predictions)
        return {
            "accuracy": correct / len(dataset),
            "exact_label_rate": exact / len(dataset),
        }

    # LoRA's B matrices initialise to zero, so this measures the base model's
    # zero-shot behaviour before any adaptation.
    model.eval()
    before = evaluate(eval_examples)
    logger.info(
        "Before fine-tuning: accuracy %.0f%%, exact-label outputs %.0f%%",
        before["accuracy"] * 100,
        before["exact_label_rate"] * 100,
    )

    def encode(item: LabelledTransaction) -> dict[str, list[int]]:
        prompt_text = format_prompt(item.description)
        full = tokenizer(
            prompt_text + item.label + tokenizer.eos_token,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        prompt_length = len(
            tokenizer(prompt_text, truncation=True, max_length=MAX_SEQUENCE_LENGTH)["input_ids"]
        )
        # Mask the prompt so loss is computed only on the label tokens.
        labels = list(full["input_ids"])
        for index in range(min(prompt_length, len(labels))):
            labels[index] = PROMPT_LABEL_MASK
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    model.train()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=args.seed,
        ),
        train_dataset=Dataset.from_list([encode(item) for item in train_examples]),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=PROMPT_LABEL_MASK, padding=True
        ),
    )
    trainer.train()

    model.eval()
    after = evaluate(eval_examples)
    logger.info(
        "After fine-tuning:  accuracy %.0f%%, exact-label outputs %.0f%%",
        after["accuracy"] * 100,
        after["exact_label_rate"] * 100,
    )

    adapter_path = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metrics = {
        "base_model": args.base_model,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
        "before": before,
        "after": after,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("Adapter saved to %s", adapter_path)
    logger.info("Use it with: AUDIT_CLASSIFIER=qlora python examples/02_vendor_review.py")


if __name__ == "__main__":
    main()
