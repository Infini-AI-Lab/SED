"""Factories for SED components used by AgentHarm runners."""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from sed.judge import Judge
from sed.synthesizer import PolicySynthesizer
from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def make_fireworks_client(api_key: str | None = None, timeout: int | None = None) -> OpenAI:
    resolved_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not resolved_key:
        raise ValueError("FIREWORKS_API_KEY is required for SED AgentHarm runs.")
    kwargs = {"api_key": resolved_key, "base_url": FIREWORKS_BASE_URL}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def build_defense_agent(
    memory_dir: str | Path,
    target_model: str,
    memory_model: str | None = None,
    synth_model: str | None = None,
    embed_model: str | None = None,
    api_key: str | None = None,
    system_prompt: str = DEFENSE_BASE_PROMPT,
    flat_memory: bool = False,
) -> DefenseAgent:
    memory_path = Path(memory_dir)
    attack_memory_dir = memory_path / "attack_memory"
    synth_log_dir = memory_path / "synth_log"
    attack_memory_dir.mkdir(parents=True, exist_ok=True)
    synth_log_dir.mkdir(parents=True, exist_ok=True)

    resolved_memory_model = memory_model or synth_model or target_model
    client = make_fireworks_client(api_key)
    synthesizer = PolicySynthesizer(
        llm_client=client,
        model=synth_model or resolved_memory_model,
        log_dir=str(synth_log_dir),
    )

    agent = DefenseAgent(
        fireworks_api_key=api_key,
        chat_model=target_model,
        memory_model=resolved_memory_model,
        embed_model=embed_model,
        policy_synthesizer=synthesizer,
        attack_memory_jsonl_l1=str(attack_memory_dir / "layer1.jsonl"),
        attack_memory_npy_l1=str(attack_memory_dir / "layer1.npy"),
        attack_memory_jsonl_l2=str(attack_memory_dir / "layer2.jsonl"),
        attack_memory_npy_l2=str(attack_memory_dir / "layer2.npy"),
        system_prompt=system_prompt,
    )
    agent.memory_manager.flat = flat_memory
    return agent


def build_judge(judge_model: str, api_key: str | None = None) -> Judge:
    return Judge(llm_client=make_fireworks_client(api_key), judge_model=judge_model,
                 parse_failure_assumes_harmful=False)

