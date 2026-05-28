import os
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv

from prompts.registry import get_prompt_registry
from tools import TOOL_REGISTRY, ToolWrapper

load_dotenv()

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")


class QAAgent:
    """Agent QA / ReAct pattern: Think → Act → Observe → Repeat."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 6,
        max_tokens: int = 4096,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.prompts = get_prompt_registry()

    def _build_system(self) -> list[dict]:
        text = self.prompts.render(
            "planner",
            role="QA expert assistant",
            domain="general factual questions",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            max_words=200,
            tool_names=sorted(TOOL_REGISTRY.keys()),
        )
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_tools(self) -> list[dict]:
        catalog = ToolWrapper.catalog()
        if catalog:
            catalog[-1] = {**catalog[-1], "cache_control": {"type": "ephemeral"}}
        return catalog

    def _extract_text(self, content: list[Any]) -> str:
        return "\n".join(block.text for block in content if block.type == "text").strip()

    def run(self, question: str, verbose: bool = False) -> str:
        messages: list[dict] = [{"role": "user", "content": question}]
        system = self._build_system()
        tools = self._build_tools()

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n=== Iteration {iteration} (Think) ===")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if verbose:
                usage = response.usage
                print(
                    f"  stop_reason={response.stop_reason} | "
                    f"input={usage.input_tokens} cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                    f"cache_write={getattr(usage, 'cache_creation_input_tokens', 0)} output={usage.output_tokens}"
                )
                preamble = self._extract_text(response.content)
                if preamble:
                    print(f"  Think: {preamble[:300]}")

            if response.stop_reason == "end_turn":
                return self._extract_text(response.content) or "(empty response)"

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return self._extract_text(response.content) or "(empty response)"

            tool_results = []
            for tool_use in tool_uses:
                if verbose:
                    print(f"  Act: {tool_use.name}({tool_use.input})")
                result = ToolWrapper.call(tool_use.name, tool_use.input)
                if verbose:
                    print(f"  Observe: {result[:200]}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"Max iterations ({self.max_iterations}) reached without final response."
        )


def main() -> None:
    agent = QAAgent()

    questions = [
        "Cât face (123 + 456) * 7, și ce dată este astăzi în zona Europe/Bucharest?",
        # "Calculează 1500 * (1 + 0.07) ** 10 și spune-mi ce companie a creat Claude AI?",
        # "Care e suma numerelor de la 1 la 100 și cine a fost Carl Friedrich Gauss?",
    ]

    for question in questions:
        print(f"\n>>> Question: {question}")
        answer = agent.run(question, verbose=True)
        print(f"\n<<< Final answer:\n{answer}\n")


if __name__ == "__main__":
    main()
