"""Anthropic Claude provider, including native tool use."""

from __future__ import annotations

import logging

from anthropic import Anthropic

from audit_agent.llm.base import ToolDefinition, ToolRequest

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Hosted Claude backend. Default for the pipeline.

    Note: no sampling temperature is sent. Current Claude models reject the
    parameter, and the workflow wants deterministic-as-possible output anyway.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or run with "
                "AUDIT_LLM_PROVIDER=ollama to use a local model."
            )
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def supports_tools(self) -> bool:
        return True

    def decide_tool_use(self, user: str, tools: list[ToolDefinition]) -> ToolRequest:
        """Ask the model whether the question warrants calling a tool.

        Letting the model make that call is the difference between a fixed
        retrieve-then-answer chain and an agent that retrieves only when the
        question actually needs evidence.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ],
            messages=[{"role": "user", "content": user}],
        )

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return ToolRequest(tool_name=None, tool_input={}, text=text)

        return ToolRequest(
            tool_name=tool_block.name,
            tool_input=dict(tool_block.input),
            text=text,
        )

    def answer_with_tool_result(
        self,
        user: str,
        tools: list[ToolDefinition],
        request: ToolRequest,
        tool_result: str,
        tool_use_id: str | None = None,
    ) -> str:
        """Second turn: hand the tool output back and get the final answer."""
        return self.complete(
            system=(
                "Answer the question using the tool output provided. Cite the source "
                "file and page for every factual claim."
            ),
            user=f"Question: {user}\n\nTool output:\n{tool_result}",
        )
