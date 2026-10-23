"""Optional LLM layer.

DriveMemory works fully offline with deterministic templates. If
``ANTHROPIC_API_KEY`` is set and the ``anthropic`` package is installed, the
orchestrator uses an LLM to phrase greetings and to answer free-form questions
grounded in the driver's memory profile.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

log = logging.getLogger(__name__)


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("DRIVEMEMORY_LLM_MODEL", "claude-haiku-5-5")

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def get_llm() -> LLM | None:
    if os.getenv("DRIVEMEMORY_DISABLE_LLM") or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicLLM()
    except Exception as e:  # package missing or misconfigured: fall back to templates
        log.warning("LLM unavailable, using templates: %s", e)
        return None
