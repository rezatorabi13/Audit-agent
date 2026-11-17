"""Multi-agent audit assistant: hybrid retrieval, graph reasoning, a QLoRA-fine-tuned
Qwen transaction classifier, and human-reviewed drafting on local Llama 3.2."""

from audit_agent.config import Settings, get_settings
from audit_agent.models import (
    AnomalyFlag,
    DocumentChunk,
    DocumentRecord,
    RelatedPartyFinding,
    RetrievedChunk,
)

__version__ = "0.1.0"

__all__ = [
    "AnomalyFlag",
    "DocumentChunk",
    "DocumentRecord",
    "RelatedPartyFinding",
    "RetrievedChunk",
    "Settings",
    "__version__",
    "get_settings",
]
