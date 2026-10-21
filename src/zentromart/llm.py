"""Minimal NVIDIA Build chat-completions client using the documented API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from .env without executing shell code."""
    env_path = path or (_project_root() / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class NVIDIACompletion:
    message: dict[str, Any]
    usage: dict[str, int]
    finish_reason: str | None


class NVIDIAAPIError(RuntimeError):
    """Safe provider error with no request body or credentials attached."""


class NVIDIAChatClient:
    """Call NVIDIA Build's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout_seconds: float = 90.0,
        rate_limit_retries: int = 0,
        rate_limit_retry_delay: float = 60.0,
    ):
        if not api_key.strip():
            raise ValueError("NVIDIA API key is required")
        if not model.strip():
            raise ValueError("NVIDIA model name is required")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_retry_delay = max(0.0, rate_limit_retry_delay)

    @classmethod
    def from_environment(cls, **client_options: Any) -> "NVIDIAChatClient":
        load_local_env()
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        model = os.environ.get("NVIDIA_MODEL", "")
        api_base = os.environ.get("NVIDIA_API_BASE", DEFAULT_API_BASE)
        if not api_key:
            raise RuntimeError("Set NVIDIA_API_KEY in the local environment or .env")
        if not model:
            raise RuntimeError("Set NVIDIA_MODEL after choosing an NVIDIA Build model")
        return cls(api_key, model, api_base=api_base, **client_options)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> NVIDIACompletion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if "glm-5.3-flash" in self.model.casefold():
            # This model defaults to max internal reasoning, which can consume
            # the whole answer budget. NVIDIA recommends clearing thinking for
            # chat; keep only a low reasoning budget for these support turns.
            body["reasoning_effort"] = "low"
            body["chat_template_kwargs"] = {"clear_thinking": True}
        elif "nemotron-3.5-lightning-30b-a3b" in self.model.casefold():
            # Keep the fast Lightning endpoint in chat mode without a long
            # internal reasoning trace, which is better suited to voice turns.
            body["chat_template_kwargs"] = {"enable_thinking": False}

        payload: dict[str, Any] | None = None
        retry_count = 0
        completed = False
        while not completed:
            request = Request(
                urljoin(f"{self.api_base}/", "chat/completions"),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    completed = True
            except HTTPError as error:
                if error.code == 429 and retry_count < self.rate_limit_retries:
                    retry_after = error.headers.get("Retry-After")
                    try:
                        delay = max(0.0, float(retry_after)) if retry_after else self.rate_limit_retry_delay
                    except ValueError:
                        delay = self.rate_limit_retry_delay
                    retry_count += 1
                    print(
                        f"NVIDIA Build returned HTTP 429; waiting {delay:g}s "
                        f"before retry {retry_count}/{self.rate_limit_retries}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                status = error.code
                hint = {
                    401: "Check NVIDIA_API_KEY.",
                    402: "NVIDIA reports insufficient API credits.",
                    404: "Check NVIDIA_MODEL and model access.",
                    429: "NVIDIA rate limit reached; retry later.",
                }.get(status, "Check model access and the request settings.")
                raise NVIDIAAPIError(
                    f"NVIDIA Build request failed (HTTP {status}). {hint}"
                ) from None
            except URLError:
                raise NVIDIAAPIError("Could not reach the NVIDIA Build API.") from None
            except TimeoutError:
                raise NVIDIAAPIError("NVIDIA Build request timed out.") from None
            except json.JSONDecodeError:
                raise NVIDIAAPIError("NVIDIA Build returned an invalid JSON response.") from None

        if not isinstance(payload, dict):
            raise NVIDIAAPIError("NVIDIA Build returned an invalid response object.")
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise NVIDIAAPIError("NVIDIA Build response did not contain a chat message.")

        usage_data = payload.get("usage") or {}
        usage = {
            key: int(usage_data.get(key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return NVIDIACompletion(
            message=choices[0]["message"],
            usage=usage,
            finish_reason=choices[0].get("finish_reason"),
        )
