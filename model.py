#!/usr/bin/env python3
"""Model setup and calling — import this to talk to the LLM from other code.

Example
-------
    from model import call_model

    reply, result = call_model([{"role": "user", "content": "hi"}])
    print(reply)
    print(result.total_tokens)
"""
import os
from functools import lru_cache
from typing import NamedTuple

from openai import OpenAI
from dotenv import load_dotenv

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


class ModelResult(NamedTuple):
    reply: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@lru_cache(maxsize=None)
def get_client() -> OpenAI:
    """Build the OpenAI-compatible client, reading credentials from the env/.env."""
    load_dotenv()
    return OpenAI(
        api_key=os.environ.get("RUNPOD_API_KEY"),
        base_url=os.environ.get("RUNPOD_BASE_URL"),
    )


def call_model(messages: list, *, model: str = MODEL, client: OpenAI | None = None) -> ModelResult:
    """Send `messages` to the model and return the reply plus token usage."""
    client = client or get_client()
    response = client.chat.completions.create(model=model, messages=messages)
    return ModelResult(
        reply=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
