"""
Anthropic Claude Provider - Native SDK implementation.

System message handling:
- Extracted from messages list and passed as separate `system` parameter
- This is Anthropic's recommended approach
"""
import logging
import os
from typing import AsyncIterator, Optional

from ..base import LLMProvider, Message

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider using native SDK.

    Environment variables:
        ANTHROPIC_API_KEY: Required API key
        ANTHROPIC_MODEL: Optional model override (default: claude-sonnet-4-20250514)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        model = model or os.getenv("ANTHROPIC_MODEL")
        super().__init__(model=model, temperature=temperature)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def is_configured(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _get_client(self):
        """Lazy initialization of Anthropic client (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                try:
                    import anthropic
                    self._client = anthropic.AsyncAnthropic()
                    logger.debug(f"Initialized Anthropic client for model {self.model}")
                except ImportError:
                    raise ImportError("anthropic package required. Install: pip install anthropic")
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """
        Extract system message and format remaining messages.

        Anthropic API expects:
        - system: str (separate parameter)
        - messages: list[{role, content}] (user/assistant only)

        Returns:
            (system_text, formatted_messages)
        """
        system = ""
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            else:
                formatted.append({"role": role, "content": content})

        return system, formatted

    async def generate(self, messages: list[Message]) -> str:
        """Generate complete response."""
        client = self._get_client()
        system, msgs = self._prepare_messages(messages)

        logger.debug(f"Anthropic generate: {len(msgs)} messages, system={bool(system)}")

        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system if system else [],
            messages=msgs,
            temperature=self.temperature,
        )

        return response.content[0].text

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response chunks."""
        client = self._get_client()
        system, msgs = self._prepare_messages(messages)

        logger.debug(f"Anthropic stream: {len(msgs)} messages, system={bool(system)}")

        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system if system else [],
            messages=msgs,
            temperature=self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Prompt caching (L8 - Task 2)
    # ------------------------------------------------------------------
    def supports_prompt_caching(self) -> bool:
        return True

    async def _generate_with_usage_async(
        self,
        messages: list[Message],
        cache_system: bool = False,
    ) -> tuple[str, dict]:
        """Variantă async a generate care întoarce și usage (cu tokeni de cache)."""
        client = self._get_client()
        system, msgs = self._prepare_messages(messages)

        # Construim `system` ca listă de blocuri. Dacă cache_system=True marcăm
        # prefixul STATIC (system) cu cache_control: ephemeral → request-urile
        # următoare cu același prefix plătesc cache_read (~0.1x) în loc de input
        # full. Min. 4096 tokeni pentru Claude Haiku 4.5; sub prag caching-ul nu
        # se activează (fără eroare, doar cache_creation_input_tokens=0).
        if system:
            system_param: list = [{"type": "text", "text": system}]
            if cache_system:
                system_param[0]["cache_control"] = {"type": "ephemeral"}
        else:
            system_param = []

        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_param,
            messages=msgs,
            temperature=self.temperature,
        )

        u = response.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            # >0 la primul apel (scrie cache, ~1.25x)
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            # >0 la apelurile următoare (citește cache, ~0.1x)
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        return response.content[0].text, usage

    def generate_with_usage(
        self,
        messages: list[Message],
        cache_system: bool = False,
    ) -> tuple[str, dict]:
        loop = self._get_sync_loop()
        return loop.run_until_complete(
            self._generate_with_usage_async(messages, cache_system=cache_system)
        )
