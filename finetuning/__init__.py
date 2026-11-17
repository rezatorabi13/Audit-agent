"""QLoRA fine-tuning of Qwen2.5-3B-Instruct as a transaction risk classifier.

Train once with ``python -m finetuning.train_qlora`` (CUDA GPU, ~8 GB VRAM), then
run the workflow with ``AUDIT_CLASSIFIER=qlora``.
"""
