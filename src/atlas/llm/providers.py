"""Provider-agnostic LLM layer: OpenAI, Azure OpenAI, Anthropic and a deterministic offline mode.

Every call goes through :func:`traced`, which records latency, token usage and a prompt hash to a
JSONL trace file — a lightweight stand-in for LangSmith-style LLMOps tracing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Protocol

from atlas.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMError(RuntimeError):
    """Raised when a provider fails after retries or declines the request."""


class LLM(Protocol):
    provider: str
    model: str
    offline: bool

    def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse: ...


class OfflineLLM:
    """No network, no key. The agent uses deterministic heuristics when ``offline`` is True."""

    provider, model, offline = "offline", "rule-based", True

    def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        return LLMResponse(text="", provider=self.provider, model=self.model)


class OpenAILLM:
    offline = False

    def __init__(self, settings: Settings, azure: bool = False):
        if azure:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.azure_api_version,
            )
            self.provider, self.model = "azure", settings.azure_deployment or ""
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=settings.openai_api_key)
            self.provider, self.model = "openai", settings.openai_model

    def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"} if json_mode else None,
        )
        usage = resp.usage
        return LLMResponse(
            resp.choices[0].message.content or "",
            self.provider,
            self.model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class AnthropicLLM:
    provider, offline = "anthropic", False

    def __init__(self, settings: Settings):
        import anthropic

        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=3)
        self.model = settings.anthropic_model

    def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        if json_mode:
            system += "\n\nRespond with a single valid JSON object and nothing else."
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Short, well-specified support tasks: low effort keeps latency and cost down.
            output_config={"effort": "low"},
            # Server-side fallback: if a safety classifier declines, the API re-runs on a
            # recommended fallback model instead of returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text, self.provider, resp.model, resp.usage.input_tokens, resp.usage.output_tokens
        )


class TracedLLM:
    """Decorator adding retries, latency/token accounting and a JSONL trace."""

    def __init__(self, inner: LLM, trace_path, retries: int = 2):
        self.inner, self.trace_path, self.retries = inner, trace_path, retries
        self.provider, self.model, self.offline = inner.provider, inner.model, inner.offline

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
        task: str = "generic",
    ) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                resp = self.inner.complete(system, user, max_tokens=max_tokens, json_mode=json_mode)
                resp.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                self._trace(task, system + user, resp, attempt)
                return resp
            except LLMError:
                raise
            except Exception as exc:  # network / rate limit → exponential backoff
                last_exc = exc
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 8))
        raise LLMError(f"{self.provider} failed after retries: {last_exc}")

    def _trace(self, task: str, prompt: str, resp: LLMResponse, attempt: int) -> None:
        if self.offline:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "task": task,
            "attempt": attempt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            **{k: v for k, v in asdict(resp).items() if k != "text"},
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def build_llm(settings: Settings | None = None) -> TracedLLM:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "openai" and settings.openai_api_key:
        inner: LLM = OpenAILLM(settings)
    elif provider == "azure" and settings.azure_api_key and settings.azure_endpoint:
        inner = OpenAILLM(settings, azure=True)
    elif provider == "anthropic" and settings.anthropic_api_key:
        inner = AnthropicLLM(settings)
    else:
        if provider != "offline":
            log.warning(
                "Provider '%s' selected but credentials missing — using offline mode.", provider
            )
        inner = OfflineLLM()
    return TracedLLM(inner, settings.artifacts_dir / "traces" / "llm_calls.jsonl")


@lru_cache
def get_llm() -> TracedLLM:
    return build_llm()
