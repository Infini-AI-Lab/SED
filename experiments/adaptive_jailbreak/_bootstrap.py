"""Make OpenRT + SED importable, and bypass unused optional deps.

OpenRT eagerly imports *every* blackbox + whitebox attack when any attack module
is loaded (see OpenRT.attacks / OpenRT.attacks.blackbox.implementations). That
pulls in heavy optional stacks (torchvision, spacy, pandas, matplotlib, …) and
even import-time side effects (Himrd raises if OPENAI_API_KEY is unset).

This experiment only needs the text black-box attackers wired in run.py:
  AutoDAN-Turbo, PAIR, TAP (TreeAttack), X-Teaming, plus DirectAttack.

We therefore:
  - put OpenRT + SED on sys.path;
  - stub `diffusers` (image path);
  - stub the entire whitebox attack package;
  - pre-register stub modules for unused blackbox attacks so their real files
    (and their deps / OPENAI checks) never execute.

Idempotent. Must be imported before any `from OpenRT...` statement.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Upstream OpenRT is not vendored — `bash setup_benchmarks.sh openrt` clones it here.
OPENRT_ROOT = Path(os.environ.get("OPENRT_ROOT", REPO_ROOT / "third_party" / "OpenRT"))

if not OPENRT_ROOT.is_dir() or not (OPENRT_ROOT / "OpenRT").is_dir():
    raise SystemExit(
        f"OpenRT not found at {OPENRT_ROOT}.\n"
        "Run:  bash setup_benchmarks.sh openrt\n"
        "or set OPENRT_ROOT to an existing OpenRT checkout."
    )

for _p in (str(REPO_ROOT), str(OPENRT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _register(name: str, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
    else:
        sys.modules[name].__dict__.update(attrs)
    return sys.modules[name]


def _stub_package(name: str, **attrs):
    """Register a package module (has __path__) so sub-imports resolve to stubs."""
    mod = _register(name, **attrs)
    # Empty path: submodules must already be registered in sys.modules.
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


# Image diffusion path (unused by this text experiment).
_register("diffusers", DiffusionPipeline=object)

# ── whitebox: matplotlib / seaborn / torchvision — unused here ───────────────
_stub_package("OpenRT.attacks.whitebox")
_register("OpenRT.attacks.whitebox.base", BaseWhiteBoxAttack=object)
_stub_package("OpenRT.attacks.whitebox.implementations", __all__=[])
_register(
    "OpenRT.attacks.whitebox.implementations.visual_jailbreak",
    VisualJailbreakAttack=object,
    VisualJailbreakConfig=object,
)
_register(
    "OpenRT.attacks.whitebox.implementations.imperceptible_jailbreak",
    ImperceptibleJailbreakAttack=object,
    ImperceptibleJailbreakConfig=object,
    ImperceptibleJailbreakResult=object,
)

# ── unused blackbox attacks (real modules have heavy / gated imports) ────────
# Names must match what OpenRT.attacks.blackbox.implementations.__init__ imports.
_UNUSED_BLACKBOX = {
    # module path relative to OpenRT.attacks.blackbox.implementations → export names
    "autodan_turbo_r": ("AutoDANTurboR",),
    "genetic_attack": ("GeneticAttack",),
    "autodan": ("AutoDAN_Attack",),
    "renellm_attack": ("ReNeLLMAttack",),
    "gptfuzzer": ("GPTFuzzerAttack",),
    "cipherchat": ("CipherChatAttack",),
    "deepinception_attack": ("DeepInceptionAttack",),
    "ica_attack": ("ICAAttack",),
    "jailbroken_attack": ("JailBrokenAttack",),
    "redqueen_attack": ("RedQueenAttack",),
    "actor_attack": ("ActorAttack",),
    "crescendo_attack": ("CrescendoAttack",),
    "flipattack": ("FlipAttack",),
    "mousetrap": ("MousetrapAttack",),
    "multilingual_attack": ("MultilingualAttack",),
    "prefill_attack": ("PrefillAttack",),
    "rainbow_teaming": ("RainbowTeamingAttack",),
    "coa": ("CoAAttack",),
    "CodeAttack": ("CodeAttack",),
    "CSDJ": ("CSDJAttack",),
    "DrAttack": ("DrAttack",),
    "FigStep": ("FigStepAttack",),
    "HADES": ("HadesAttack",),
    "himrd": ("HimrdAttack",),
    "IDEATOR": ("IdeatorAttack",),
    "jood": ("JoodAttack",),
    "MML": ("MMLAttack",),
    "query_relevant": ("QueryRelevantAttack",),
    "race": ("RACEAttack",),
}

_BB = "OpenRT.attacks.blackbox.implementations"
for _mod, _exports in _UNUSED_BLACKBOX.items():
    _attrs = {name: object for name in _exports}
    _stub_package(f"{_BB}.{_mod}", **_attrs)
    # Some packages re-export from a nested attack.py; cover that path too.
    _register(f"{_BB}.{_mod}.attack", **_attrs)

# Evosynth (openai-agents) — keep the previous short-circuit.
_EVO = f"{_BB}.evosynth"
_stub_package(_EVO)
_register(
    f"{_EVO}.evosynth_attack",
    EvosynthAttack=object,
    EvosynthConfig=object,
    create_evosynth_attack=lambda *a, **k: None,
)
