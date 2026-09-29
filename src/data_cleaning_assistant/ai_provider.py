from __future__ import annotations

import json
import os
import socket
from typing import Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class AIProviderError(RuntimeError):
    """Raised when a local or hosted AI provider cannot fulfill a request."""


class StructuredAIProvider(Protocol):
    name: str

    def list_models(self) -> list[str]:
        ...

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        ...


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 180.0,
        think: bool = False,
    ) -> None:
        configured = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        if "://" not in configured:
            configured = f"http://{configured}"
        self.base_url = configured.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.think = think

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AIProviderError(
                f"Ollama returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except TimeoutError as exc:
            raise AIProviderError(
                f"Ollama request to {path} timed out after "
                f"{self.timeout_seconds:g} seconds."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise AIProviderError(
                    f"Ollama request to {path} timed out after "
                    f"{self.timeout_seconds:g} seconds."
                ) from exc
            raise AIProviderError(
                f"Could not reach Ollama at {self.base_url}: {exc.reason}. "
                "Check that the local Ollama service is running and reachable."
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AIProviderError("Ollama returned a non-JSON response.") from exc
        if not isinstance(parsed, dict):
            raise AIProviderError("Ollama returned an unexpected response shape.")
        return parsed

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        output: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name:
                output.append(name)
        return sorted(set(output))

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        payload = self._request(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": self.think,
                "format": response_model.model_json_schema(),
                "options": {"temperature": 0.1},
            },
        )

        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise AIProviderError("Ollama response did not contain message.content.")

        try:
            return response_model.model_validate_json(message["content"])
        except ValidationError as exc:
            raise AIProviderError(
                "Ollama returned structured output that failed local schema validation."
            ) from exc
