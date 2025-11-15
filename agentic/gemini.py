"""Gemini API client with offline fallbacks."""

from __future__ import annotations

import json
import os
import random
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GeminiResponse:
    """Simple container for the model response."""

    text: str
    model: str
    prompt_tokens: int = 0
    candidate_tokens: int = 0


class GeminiClient:
    """Minimal client for the Gemini REST API.

    The implementation uses urllib to avoid third-party dependencies.
    When an API key is missing the client falls back to a deterministic
    offline response so that unit tests can run in constrained sandboxes.
    """

    api_url_template = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        env_model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL")
        self.model = self._normalize_model_name(model or env_model)
        self.temperature = temperature

    def generate(self, prompt: str, system_instruction: str | None = None) -> GeminiResponse:
        if not self.api_key:
            return self._offline_response(prompt, system_instruction)

        url = self.api_url_template.format(model=self.model)
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": self._compose_prompt(prompt, system_instruction)}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{url}?key={self.api_key}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on network
            raise RuntimeError(f"Gemini API error: {exc.read().decode('utf-8', 'ignore')}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - depends on network
            raise RuntimeError(f"Gemini API unreachable: {exc}") from exc

        candidates = body.get("candidates", [])
        if not candidates:
            return GeminiResponse(text="Gemini API returned no candidates.", model=self.model)

        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts]
        return GeminiResponse(
            text="\n".join(filter(None, text_parts)),
            model=self.model,
            prompt_tokens=body.get("usageMetadata", {}).get("promptTokenCount", 0),
            candidate_tokens=body.get("usageMetadata", {}).get("candidatesTokenCount", 0),
        )

    def _compose_prompt(self, prompt: str, system_instruction: str | None) -> str:
        if system_instruction:
            return textwrap.dedent(
                f"""{system_instruction.strip()}

                User request: {prompt.strip()}
                """
            ).strip()
        return prompt

    def _normalize_model_name(self, model: str | None) -> str:
        """Map shorthand labels to the full Gemini REST identifiers."""

        if not model:
            return "models/gemini-1.5-flash"

        trimmed = model.strip()
        shorthand_map = {
            "gemini-2.5-flash": "models/gemini-2.0-flash-exp",
            "gemini-2.0-flash": "models/gemini-2.0-flash-exp",
            "2.5-flash": "models/gemini-2.0-flash-exp",
            "2.0-flash": "models/gemini-2.0-flash-exp",
            "1.5-flash": "models/gemini-1.5-flash",
        }

        if trimmed in shorthand_map:
            return shorthand_map[trimmed]

        if not trimmed.startswith("models/"):
            return f"models/{trimmed}"

        return trimmed

    def _offline_response(self, prompt: str, system_instruction: str | None) -> GeminiResponse:
        seed = hash((prompt, system_instruction, self.model)) & 0xFFFF
        random.seed(seed)
        templates = [
            "(offline) Analyzed the scenario and highlighted the key vitals.",
            "(offline) Summarized patient intent and pulled safety watch-outs.",
            "(offline) Crafted a cautious response referencing internal knowledge.",
        ]
        choice = random.choice(templates)
        scaffold = textwrap.dedent(
            f"""{choice}

            Prompt excerpt: {prompt[:180].strip()}..."""
        )
        return GeminiResponse(text=scaffold, model=self.model)
