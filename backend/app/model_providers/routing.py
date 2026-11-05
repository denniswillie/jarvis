"""Selectable routing inference; never changes the conversational audio client."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from openai import AsyncOpenAI


META_BASE_URL = "https://api.meta.ai/v1"


def meta_api_key(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    raw_path = env.get("JARVIS_META_API_KEY_FILE", "")
    path = Path(raw_path)
    if not raw_path or not path.is_absolute() or not path.is_file():
        raise ValueError("JARVIS_META_API_KEY_FILE must name an existing absolute file")
    key = path.read_text(encoding="utf-8").strip()
    if not key or key.startswith("REPLACE_"):
        raise ValueError("Meta API key is not configured")
    return key


@dataclass(frozen=True)
class RouterModelConfiguration:
    model: str = "gpt-6-luna"
    reasoning_effort: str = "none"
    tool_choice: str = "required"
    max_output_tokens: int = 256
    structured_selection: bool = False

    def parameters(self) -> dict:
        return {"model": self.model, "reasoning": {"effort": self.reasoning_effort},
                "tool_choice": self.tool_choice, "max_output_tokens": self.max_output_tokens}


async def create_router_selection(client, configuration, **request):
    if configuration.structured_selection:
        from app.model_providers.structured_router import create_structured_selection
        return await create_structured_selection(client, **request)
    return await client.responses.create(**request)


def create_router_client(settings, *, environ: Mapping[str, str] | None = None):
    env = os.environ if environ is None else environ
    provider = env.get("JARVIS_ROUTER_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        # Explicit endpoint: a Meta worker's OPENAI_BASE_URL must not redirect Luna.
        return (AsyncOpenAI(api_key=settings.OPENAI_API_KEY,
                            base_url="https://api.openai.com/v1", max_retries=0),
                RouterModelConfiguration())
    if provider != "meta":
        raise ValueError("JARVIS_ROUTER_PROVIDER must be openai or meta")
    config = RouterModelConfiguration(
        model=env.get("JARVIS_META_ROUTER_MODEL", "muse-spark-1.3"),
        reasoning_effort="minimal", tool_choice="auto", max_output_tokens=2048,
        structured_selection=True)
    return (AsyncOpenAI(api_key=meta_api_key(env), base_url=META_BASE_URL, max_retries=0), config)
