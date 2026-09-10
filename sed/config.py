"""Default SED hyperparameters — the single place to change a knob.

Symbols match the paper's hyperparameter table (Appendix: Hyperparameters and
Notation). Nothing here imports from the rest of `sed`, so it stays at the
bottom of the dependency graph.
"""

# ─── Policy-tree placement ────────────────────────────────────────────────────

K_INJECT = 3          # K          policies injected per query
FAMILY_CAP = 2        # c          per-family cap on injection (0 = off)
PARENT_PULL = True    #            also inject each leaf's nearest covering parent
K_PLACEMENT = 5       # k          nearest policy names considered at placement
TAU_NEW = 0.55        # tau_new    best name similarity below this -> new rule at root
TAU_MERGE = 0.80      # tau_merge  name similarity at/above this -> same node
SYNTH_BATCH = 1       # b          harmful episodes buffered before synthesis fires
MAX_NEW_POLICIES = 3  # c_max      policies proposed per synthesis call

# ─── Models ───────────────────────────────────────────────────────────────────

EMBED_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_EMBED_URL = "https://api.fireworks.ai/inference/v1/embeddings"

# ─── Generation limits ────────────────────────────────────────────────────────

JUDGE_MAX_TOKENS = 4096      # judge verdict JSON
ORGANIZER_MAX_TOKENS = 8192  # placement: reasoning block + JSON; too low silently
                             # truncates the decision into ADD(root)

# ─── Trajectory handling ──────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 20  # conversation turns retained in the agent's context
TRUNCATE_TAIL = 200     # chars kept from the tail when truncating agent responses
