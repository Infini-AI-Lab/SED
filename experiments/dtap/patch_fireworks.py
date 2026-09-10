#!/usr/bin/env python3
"""Patch DecodingTrust-Agent OpenAI SDK backend for Fireworks (OpenAI-compatible API).

The OpenAI Agents SDK rejects Fireworks model IDs (accounts/fireworks/models/...)
unless we route through OpenAIChatCompletionsModel with a custom base_url — same
pattern DTap already uses for OpenRouter.
"""
from __future__ import annotations

import sys
from pathlib import Path

AGENT_PY = Path("agent/openaisdk/src/agent.py")

HELPER = '''
def _chat_completions_base_url() -> str:
    """Return a custom OpenAI-compatible base URL if one is configured."""
    for key in ("OPENAI_BASE_URL", "FIREWORKS_BASE_URL"):
        url = os.environ.get(key, "").strip()
        if url:
            return url
    return ""


def _use_chat_completions_api(base_url: str) -> bool:
    if not base_url:
        return False
    return any(host in base_url for host in ("openrouter.ai", "fireworks.ai"))


def _make_chat_completions_model(model_name: str):
    base_url = _chat_completions_base_url()
    if not _use_chat_completions_api(base_url):
        return model_name
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    return OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(base_url=base_url, api_key=api_key),
    )
'''

OLD_INIT = '''        # Determine model: use Chat Completions API for OpenRouter,
        # since it only supports the Chat Completions format, not the
        # Responses API that the SDK uses by default.
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url and "openrouter.ai" in base_url:
            model = OpenAIChatCompletionsModel(
                model=self.runtime_config.model,
                openai_client=AsyncOpenAI(
                    base_url=base_url,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                ),
            )
        else:
            model = self.runtime_config.model'''

NEW_INIT = '''        model = _make_chat_completions_model(self.runtime_config.model)'''

OLD_SUB = '''        _sub_model = os.getenv("AGENT_MODEL", self.runtime_config.model)
        _sub_temp = (
            self.runtime_config.temperature
            if self.runtime_config.temperature is not None
            and _model_supports_temperature(_sub_model)
            else None
        )
        return OpenAIAgent(
            name=sub_config.name,
            instructions=sub_config.system_prompt,
            model=_sub_model,'''

NEW_SUB = '''        _sub_model_name = os.getenv("AGENT_MODEL", self.runtime_config.model)
        _sub_model = _make_chat_completions_model(_sub_model_name)
        _sub_temp = (
            self.runtime_config.temperature
            if self.runtime_config.temperature is not None
            and _model_supports_temperature(_sub_model_name)
            else None
        )
        return OpenAIAgent(
            name=sub_config.name,
            instructions=sub_config.system_prompt,
            model=_sub_model,'''


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    path = root / AGENT_PY
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    if "_make_chat_completions_model" in text:
        print("already patched:", path)
        return 0
    anchor = "def _model_supports_temperature(model_name: str) -> bool:"
    if anchor not in text:
        print("unexpected agent.py layout", file=sys.stderr)
        return 1
    text = text.replace(anchor, HELPER + "\n\n" + anchor, 1)
    if OLD_INIT not in text:
        print("initialize() block not found", file=sys.stderr)
        return 1
    text = text.replace(OLD_INIT, NEW_INIT, 1)
    if OLD_SUB not in text:
        print("subagent block not found", file=sys.stderr)
        return 1
    text = text.replace(OLD_SUB, NEW_SUB, 1)
    path.write_text(text, encoding="utf-8")
    print("patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
