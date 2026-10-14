"""Remote LA and VA providers, explicit timeouts and no automatic SDK retries."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .config import LLMConfig

log = logging.getLogger("lavira_m3.llm")

# The OpenAI SDK is imported lazily so that offline tools and tests that never
# build a router do not require it. Tests may patch this module attribute.
OpenAI = None


def _openai_cls():
    global OpenAI
    if OpenAI is None:
        from openai import OpenAI as _OpenAI

        OpenAI = _OpenAI
    return OpenAI


class LLMRouter:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        if self.cfg.la_backend == "deepseek" and not self.cfg.la_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is empty")
        if self.cfg.va_backend == "deepseek" and not self.cfg.va_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is empty")
        client_cls = _openai_cls()
        self.la = client_cls(
            api_key=self.cfg.la_api_key or "sk-no-key-required",
            base_url=self.cfg.la_base_url,
            timeout=180,
            max_retries=0,
        )
        if (
            self.cfg.va_backend == self.cfg.la_backend
            and self.cfg.va_base_url == self.cfg.la_base_url
            and self.cfg.va_api_key == self.cfg.la_api_key
        ):
            self.va = self.la
        else:
            self.va = client_cls(
                api_key=self.cfg.va_api_key,
                base_url=self.cfg.va_base_url,
                timeout=180,
                max_retries=0,
            )
        self.call_logger = None
        self._pending_record = None

    def bind_call_logger(self, logger) -> None:
        self.call_logger = logger

    def note_record(self, kind: str, step: Optional[int] = None) -> None:
        self._pending_record = (str(kind), step)

    def _emit_record(
        self, default_kind: str, text: str, duration_ms: float, usage: Any, model: str, backend: str
    ) -> None:
        pending = self._pending_record
        self._pending_record = None
        if pending is None or self.call_logger is None:
            return
        kind, step = pending
        try:
            self.call_logger.write(
                kind=kind or default_kind,
                raw=text,
                returned_at=time.time(),
                duration_ms=duration_ms,
                usage=usage,
                model=model,
                backend=backend,
                step=step,
            )
        except Exception:
            log.exception("llm call log failed kind=%s", kind)

    def generate_la(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        if self.cfg.la_backend in ("local", "aliyun"):
            extra = {"enable_thinking": False}
        else:
            extra = {"thinking": {"type": "disabled"}}
        token_kw = (
            {"max_tokens": max_tokens}
            if self.cfg.la_backend in ("local", "aliyun")
            else {"max_completion_tokens": max_tokens}
        )
        started = time.perf_counter()
        resp = self.la.chat.completions.create(
            model=self.cfg.la_model,
            messages=messages,
            extra_body=extra,
            **token_kw,
        )
        text = resp.choices[0].message.content or ""
        duration_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(resp, "usage", None)
        log.info("LA tokens=%s", usage)
        self._emit_record("la", text, duration_ms, usage, self.cfg.la_model, self.cfg.la_backend)
        return text

    def generate_va(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        if self.cfg.va_backend == "aliyun":
            extra = {"enable_thinking": False}
            token_kw = {"max_tokens": max_tokens}
        elif self.cfg.va_backend == "local":
            extra = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            token_kw = {"max_tokens": max_tokens}
        else:
            extra = {"thinking": {"type": "disabled"}}
            token_kw = {"max_completion_tokens": max_tokens}
        started = time.perf_counter()
        resp = self.va.chat.completions.create(
            model=self.cfg.va_model,
            messages=messages,
            temperature=0.0,
            extra_body=extra,
            **token_kw,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        if not text:
            text = getattr(msg, "reasoning_content", None) or ""
        duration_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(resp, "usage", None)
        finish = getattr(resp.choices[0], "finish_reason", None)
        log.info(
            "VA backend=%s model=%s tokens=%s finish=%s content_len=%s",
            self.cfg.va_backend,
            self.cfg.va_model,
            usage,
            finish,
            len(text),
        )
        self._emit_record("va", text, duration_ms, usage, self.cfg.va_model, self.cfg.va_backend)
        return text
