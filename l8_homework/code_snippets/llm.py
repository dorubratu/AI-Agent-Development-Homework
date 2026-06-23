"""
llm.py — the single import every demo in this lecture uses.

Design goal: a demo should start with ONE line and no async, no registry,
no provider plumbing visible. All the agnostic/production machinery you
already have stays where it belongs; this file is just the teaching facade.

    from llm import chat
    print(chat("Hello"))                       # -> str
    print(chat([{"role": "user", "content": "Hi"}]))

It talks to Anthropic directly (sync) so demos actually run. Everything
returns plain strings and plain dicts — nothing for a student to learn
before they learn the lecture's actual topic.

Env:
    ANTHROPIC_API_KEY   required
    ANTHROPIC_MODEL     optional (default below)
"""

from __future__ import annotations
from dotenv import load_dotenv
import os
from typing import Union

import anthropic

load_dotenv(override=True)

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# One shared client. Created lazily so importing this file never fails
# just because a key is missing (handy when only showing snippets).
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


# A message is the same shape the Anthropic API wants: {role, content}.
# content may be a plain string OR a list of blocks (needed later for
# prompt caching with cache_control). We don't constrain it — keeps demos honest.
Messages = list[dict]


def _normalize(prompt: Union[str, Messages]) -> Messages:
    """Accept either a bare string or a full messages list."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def chat(
    prompt: Union[str, Messages],
    *,
    system: Union[str, list, None] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    return_usage: bool = False,
):
    """
    Send a prompt, get back text. Sync, blocking, one call.

    Args:
        prompt: a string, or a messages list [{role, content}, ...]
        system: optional system prompt (str, or list of blocks for caching)
        return_usage: if True, return (text, usage_dict) so caching demos
                      can show cache_read / cache_creation token counts.

    Returns:
        str, or (str, dict) when return_usage=True
    """
    client = _get_client()
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=_normalize(prompt),
    )
    if system is not None:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)
    text = resp.content[0].text

    if return_usage:
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            # present only when prompt caching is in play; default to 0
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        return text, usage

    return text


def embed(texts: Union[str, list[str]]) -> list[list[float]]:
    """
    Tiny embedding helper for the caching demos (semantic cache, embedding
    cache). Anthropic has no first-party embeddings endpoint, so we use
    sentence-transformers locally — zero extra API keys, runs offline.

        pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer

    global _embed_model
    try:
        _embed_model
    except NameError:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if isinstance(texts, str):
        texts = [texts]
    return _embed_model.encode(texts, normalize_embeddings=True).tolist()


if __name__ == "__main__":
    # Smoke test: python llm.py
    print(chat("Say 'memory and caching' and nothing else."))
