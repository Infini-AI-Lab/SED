"""Prefix forcing for the AutoDAN-Turbo attacker.

OpenRT's attacker builds a `condition` string ending in `[START OF JAILBREAK
PROMPT]` but never uses it, and sends the raw chat output to the target — so the
attacker's own preamble ("**JAILBREAK PROMPT...**", markdown headers, etc.) leaks
into the prompt, making it trivial for a guard and weaker overall.

`conditional_generate` does true prefix forcing via /v1/completions: it prefills
the assistant turn with `condition`, so the model can only generate the tokens
that follow `[START OF JAILBREAK PROMPT]` — no preamble, no refusal-restart.

The chat template must match the attacker model. Default is DeepSeek (the default
attacker, deepseek-v4-flash); swap `chat_template` for a different model family.
"""
from __future__ import annotations

from OpenRT.models.implementations.openai_model import OpenAIModel

# DeepSeek V3/V4 chat template (special tokens use full-width pipe U+FF5C).
DEEPSEEK_TEMPLATE = "<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{condition}"
DEEPSEEK_STOP = ["<｜end▁of▁sentence｜>", "<｜User｜>", "[END OF JAILBREAK PROMPT]"]


class PrefixForcingModel(OpenAIModel):
    def __init__(self, *args, chat_template: str = DEEPSEEK_TEMPLATE,
                 stop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_template = chat_template
        self.cg_stop = stop if stop is not None else DEEPSEEK_STOP

    def conditional_generate(self, condition: str, system: str, user: str, **kwargs) -> str:
        prompt = self.chat_template.format(system=system, user=user, condition=condition)
        params = dict(model=self.model_name, prompt=prompt, max_tokens=self.max_tokens or 4096,
                      temperature=self.temperature if self.temperature is not None else 1.0,
                      stop=self.cg_stop)
        seed = self.chat_kwargs.get("seed")
        if seed is not None:
            params["seed"] = seed
        for attempt in range(self.retry_attempts):
            try:
                resp = self.client.completions.create(**params)
                return resp.choices[0].text or ""
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    return f"Error: {e}"
        return ""
