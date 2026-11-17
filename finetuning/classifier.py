"""The fine-tuned Qwen + QLoRA classifier, implementing the workflow's protocol.

Because it satisfies ``TransactionClassifier``, it substitutes for the default
rule-based labeller without any change to the graph. The workflow builds it
automatically when ``AUDIT_CLASSIFIER=qlora``; see
``audit_agent.pipeline.build_classifier``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from audit_agent.workflow.classifier import CATEGORIES

from finetuning.dataset import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 12
ADAPTER_CONFIG = "adapter_config.json"


class AdapterNotFoundError(FileNotFoundError):
    """The QLoRA adapter has not been trained yet on this machine."""


def ensure_adapter_exists(adapter_path: str | Path) -> Path:
    """Fail early, with the command that fixes it, if no adapter is present."""
    path = Path(adapter_path)
    if not (path / ADAPTER_CONFIG).is_file():
        raise AdapterNotFoundError(
            f"No trained QLoRA adapter at '{path}'. The adapter is not shipped with "
            "the repository; train it once (CUDA GPU, ~8 GB VRAM) with:\n"
            "    python -m finetuning.train_qlora\n"
            "or run with AUDIT_CLASSIFIER=rule_based."
        )
    return path


class LoRAClassifier:
    """Loads a trained LoRA adapter over its 4-bit quantised Qwen base model."""

    name = "qlora"

    def __init__(
        self,
        adapter_path: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        load_in_4bit: bool = True,
    ) -> None:
        adapter_path = ensure_adapter_exists(adapter_path)

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "The fine-tuned classifier requires torch, transformers, peft and "
                "bitsandbytes: pip install -r requirements.txt"
            ) from exc

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            if load_in_4bit
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        self._model = PeftModel.from_pretrained(model, str(adapter_path))
        self._model.eval()
        logger.info("Loaded QLoRA adapter %s over %s", adapter_path, base_model)

    def classify(self, description: str) -> str:
        """Return one category label, snapped to the fixed vocabulary."""
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transaction: {description}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return snap_to_vocabulary(generated.strip())


def snap_to_vocabulary(raw: str) -> str:
    """Map a raw generation onto a known category.

    Fine-tuning makes the output format reliable but not guaranteed; downstream
    code parses these labels, so an unrecognised string must not leak through.
    """
    lowered = raw.lower()
    for category in CATEGORIES:
        if category.lower() in lowered:
            return category
    logger.warning("Unrecognised classifier output %r; defaulting to Normal", raw)
    return "Normal"
