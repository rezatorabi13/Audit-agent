"""Train a 4-bit QLoRA adapter for transaction risk classification.

Run from the repository root:

    python -m finetuning.train_qlora --output-dir data/models/risk_classifier

Reports accuracy before and after training on a held-out split, so the adapter's
contribution is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from audit_agent.logging_config import configure_logging

from finetuning.dataset import (
    SYSTEM_PROMPT,
    LabelledTransaction,
    generate_examples,
    train_eval_split,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQUENCE_LENGTH = 256
PROMPT_LABEL_MASK = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/risk_classifier"))
    parser.add_argument("--examples-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

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

    from audit_agent.workflow.classifier import CATEGORIES

    examples = generate_examples(per_class=args.examples_per_class, seed=args.seed)
    train_examples, eval_examples = train_eval_split(examples)
    logger.info("%d training / %d evaluation examples", len(train_examples), len(eval_examples))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def accuracy(dataset: list[LabelledTransaction]) -> float:
        correct = sum(item.label.lower() in predict(item.description).lower() for item in dataset)
        return correct / len(dataset)

    # LoRA's B matrices initialise to zero, so this measures the base model's
    # zero-shot behaviour before any adaptation.
    before = accuracy(eval_examples)
    logger.info("Accuracy before fine-tuning: %.0f%%", before * 100)

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
            bf16=True,
            seed=args.seed,
        ),
        train_dataset=Dataset.from_list([encode(item) for item in train_examples]),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=PROMPT_LABEL_MASK, padding=True
        ),
    )
    trainer.train()

    after = accuracy(eval_examples)
    logger.info("Accuracy before: %.0f%% | after: %.0f%%", before * 100, after * 100)

    off_vocabulary = sum(
        predict(item.description) not in CATEGORIES for item in eval_examples
    )
    logger.info(
        "Off-vocabulary outputs after training: %d/%d", off_vocabulary, len(eval_examples)
    )

    adapter_path = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    logger.info("Adapter saved to %s", adapter_path)


if __name__ == "__main__":
    main()
