"""Local-first defaults, Ollama tool calling, and the QLoRA classifier plumbing.

None of these tests need a GPU, torch, an Ollama server, or a trained adapter.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from audit_agent.config import Settings
from audit_agent.knowledge.entity_graph import EntityGraph
from audit_agent.llm.base import ToolDefinition
from audit_agent.llm.ollama_provider import OllamaProvider
from audit_agent.qa import SEARCH_TOOL
from audit_agent.retrieval.hybrid import HybridRetriever
from audit_agent.workflow.classifier import CATEGORIES, describe_transaction
from audit_agent.workflow.graph import AuditWorkflow
from audit_agent.workflow.nodes import WorkflowDependencies
from finetuning.classifier import AdapterNotFoundError, ensure_adapter_exists, snap_to_vocabulary
from finetuning.dataset import generate_examples, train_eval_split

from .conftest import CLIENT, RELATED_VENDOR, StubLLM, StubVectorStore

# --- Settings ---------------------------------------------------------------


def test_defaults_are_local_llama_and_rule_based() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.classifier == "rule_based"
    assert settings.qlora_base_model == "Qwen/Qwen2.5-3B-Instruct"


def test_adapter_path_is_the_training_scripts_output() -> None:
    settings = Settings(_env_file=None, data_dir=Path("somewhere"))  # type: ignore[call-arg]

    assert settings.qlora_adapter_dir == Path("somewhere/models/qwen_risk_classifier/adapter")


# --- Ollama tool calling ----------------------------------------------------


class FakeOllama:
    """Mimics ``ollama.chat`` so the provider can be tested without a server."""

    def __init__(self, message: dict[str, Any]) -> None:
        self._message = message
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"message": self._message}


def test_ollama_supports_tool_use() -> None:
    assert OllamaProvider(client=FakeOllama({"content": ""})).supports_tools()


def test_ollama_returns_the_requested_tool_call() -> None:
    fake = FakeOllama(
        {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_audit_documents",
                        "arguments": {"query": "Globex termination"},
                    }
                }
            ],
        }
    )
    request = OllamaProvider(client=fake).decide_tool_use("Termination terms?", [SEARCH_TOOL])

    assert request.wants_tool
    assert request.tool_name == "search_audit_documents"
    assert request.tool_input == {"query": "Globex termination"}
    sent_tool = fake.calls[0]["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["function"]["parameters"] == SEARCH_TOOL.input_schema


def test_ollama_answers_directly_when_no_tool_is_called() -> None:
    fake = FakeOllama({"content": "15% of 200 is 30."})
    request = OllamaProvider(client=fake).decide_tool_use("What is 15% of 200?", [SEARCH_TOOL])

    assert not request.wants_tool
    assert request.text == "15% of 200 is 30."


def test_ollama_ignores_unknown_tools_and_parses_json_arguments() -> None:
    message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name="delete_everything", arguments={})),
            SimpleNamespace(
                function=SimpleNamespace(name="search_audit_documents", arguments='{"query": "x"}')
            ),
        ],
    )
    fake = SimpleNamespace(chat=lambda **_: SimpleNamespace(message=message))
    tools = [SEARCH_TOOL, ToolDefinition("other", "unused", {"type": "object"})]

    request = OllamaProvider(client=fake).decide_tool_use("q", tools)

    assert request.tool_name == "search_audit_documents"
    assert request.tool_input == {"query": "x"}


# --- QLoRA classifier plumbing ----------------------------------------------


def test_missing_adapter_fails_with_the_training_command(tmp_path: Path) -> None:
    with pytest.raises(AdapterNotFoundError, match="python -m finetuning.train_qlora"):
        ensure_adapter_exists(tmp_path / "adapter")


def test_existing_adapter_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    assert ensure_adapter_exists(tmp_path) == tmp_path


def test_build_classifier_follows_the_setting(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from audit_agent.pipeline import build_classifier
    from audit_agent.workflow.classifier import RuleBasedClassifier

    assert isinstance(build_classifier(Settings(_env_file=None)), RuleBasedClassifier)  # type: ignore[call-arg]

    qlora = Settings(_env_file=None, classifier="qlora", data_dir=tmp_path)  # type: ignore[call-arg]
    with pytest.raises(AdapterNotFoundError):
        build_classifier(qlora)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Duplicate Payment Risk", "Duplicate Payment Risk"),
        ("related party", "Related Party"),
        ("Label: Unusual Timing or Amount.", "Unusual Timing or Amount"),
        ("I am not sure", "Normal"),
    ],
)
def test_generations_snap_to_the_vocabulary(raw: str, expected: str) -> None:
    assert snap_to_vocabulary(raw) == expected


# --- Train/serve description format -----------------------------------------


def test_description_carries_weekday_and_ledger_wide_facts() -> None:
    text = describe_transaction(
        vendor="Acme",
        amount=50000,
        posted_on="2025-06-01",
        memo="Packaging film",
        shared_directors=("Jane Whitfield",),
        appears_twice=True,
    )

    assert "Sunday" in text
    assert "$50,000.00" in text
    assert "Jane Whitfield" in text
    assert "same amount appears in the ledger" in text


def test_plain_description_mentions_no_risk_facts() -> None:
    text = describe_transaction(vendor="Globex", amount=8200.5, posted_on="2025-04-02")

    assert "Wednesday" in text
    assert "board member" not in text
    assert "appears in the ledger" not in text


def test_training_data_is_balanced_seeded_and_in_vocabulary() -> None:
    examples = generate_examples(per_class=10, seed=3)

    assert examples == generate_examples(per_class=10, seed=3)
    assert {item.label for item in examples} == set(CATEGORIES)
    assert all(sum(e.label == c for e in examples) == 10 for c in CATEGORIES)

    train, evaluation = train_eval_split(examples)
    assert len(train) == 32 and len(evaluation) == 8


# --- A model-based classifier inside the workflow ---------------------------


class RecordingClassifier:
    """Stands in for LoRAClassifier: records what the node sends it."""

    name = "recording"

    def __init__(self) -> None:
        self.descriptions: list[str] = []

    def classify(self, description: str) -> str:
        self.descriptions.append(description)
        return "Related Party"


def test_model_classifier_receives_graph_and_duplicate_facts(
    ledger: pd.DataFrame, related_parties: pd.DataFrame
) -> None:
    classifier = RecordingClassifier()
    deps = WorkflowDependencies(
        retriever=HybridRetriever(
            vector_store=StubVectorStore(),  # type: ignore[arg-type]
            entity_graph=EntityGraph.from_dataframes(CLIENT, related_parties, ledger),
            ledger=ledger,
        ),
        llm=StubLLM(),
        ledger=ledger,
        client_name=CLIENT,
        classifier=classifier,
    )
    workflow = AuditWorkflow(deps)

    pending = workflow.start(RELATED_VENDOR, "Assess risk.")
    labels = workflow.state(pending.thread_id)["transaction_labels"]

    assert len(classifier.descriptions) == 3
    assert all("Jane Whitfield" in text for text in classifier.descriptions)
    assert sum("same amount appears" in text for text in classifier.descriptions) == 2
    assert all(label.endswith("Related Party") for label in labels)
