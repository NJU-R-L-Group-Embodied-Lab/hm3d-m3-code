"""Hosted VLM credentials and endpoint. Never hard-code keys.

All values come from the environment:

* ``LLM_API_KEY``  -- API key (fallbacks: ``DASHSCOPE_API_KEY``, ``LAVIRA_API_KEY``)
* ``LLM_BASE_URL`` -- OpenAI-compatible endpoint (fallback: ``DASHSCOPE_BASE_URL``;
  default: Aliyun Bailian / DashScope compatible mode)
* ``LLM_MODEL``    -- model name (default ``qwen3.5-27b``); used by every VLM call
  (LA/VA and Verifier V2) so they stay in sync

The paper results were produced with Qwen3.5-27B on DashScope compatible mode.
"""
from __future__ import annotations

import os
from typing import Tuple

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-27b"

API_KEY_ENV = ("LLM_API_KEY", "DASHSCOPE_API_KEY", "LAVIRA_API_KEY")
BASE_URL_ENV = ("LLM_BASE_URL", "DASHSCOPE_BASE_URL")
MODEL_ENV = ("LLM_MODEL",)


def _first_env(names) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def get_llm_api_key() -> str:
    """API key from ``LLM_API_KEY`` (then ``DASHSCOPE_API_KEY``, ``LAVIRA_API_KEY``); '' if unset."""
    return _first_env(API_KEY_ENV)


def get_llm_base_url() -> str:
    """Endpoint from ``LLM_BASE_URL`` (then ``DASHSCOPE_BASE_URL``), else the DashScope default."""
    return _first_env(BASE_URL_ENV) or DEFAULT_BASE_URL


def get_llm_model() -> str:
    """Model name from ``LLM_MODEL``, else ``qwen3.5-27b``."""
    return _first_env(MODEL_ENV) or DEFAULT_MODEL


def require_llm() -> Tuple[str, str]:
    """Return ``(base_url, api_key)`` or raise if no key is configured."""
    key = get_llm_api_key()
    if not key:
        raise RuntimeError(
            "Set LLM_API_KEY (or DASHSCOPE_API_KEY) for the OpenAI-compatible VLM endpoint "
            "(default: Aliyun Bailian / DashScope compatible mode)."
        )
    return get_llm_base_url(), key


# Backwards-compatible names used elsewhere in the code base.
def dashscope_url() -> str:
    return get_llm_base_url()


def dashscope_key() -> str:
    return get_llm_api_key()


def require_dashscope() -> Tuple[str, str]:
    return require_llm()
