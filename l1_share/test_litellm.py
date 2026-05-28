#!/usr/bin/env python3
"""Simple script to test LiteLLM API calls using httpx directly."""

import asyncio
import httpx

BASE_URL = "http://localhost:4000"
MODEL = "gemini"
SYSTEM_PROMPT = "You are a helpful assistant. Be concise."
MESSAGE = "Hello! Respond in one sentence."

def is_litellm_running() -> bool:
    """Check if LiteLLM server is running."""
    try:
        response = httpx.get(f"{BASE_URL}/health/liveliness", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


async def chat(message: str) -> str:
    """Send a chat message and return the response."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def main():
    print(f"LiteLLM available: {is_litellm_running()}")
    print(f"Model: {MODEL}")
    print(f"System: {SYSTEM_PROMPT}")
    print("\nTesting chat...")
    response = await chat(MESSAGE)
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
