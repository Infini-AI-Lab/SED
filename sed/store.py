"""
Memory Store
===========
Flat JSONL persistence + numpy cosine-similarity retrieval.
Borrowed retrieval pattern from Memento/memory/np_memory.py.

Each memory entry on disk:
  {"id": str, "content": str, "timestamp": str, "metadata": dict}

  When a content_fn is provided at construction, "content" is omitted from the JSONL
  and the embedding text is reconstructed from metadata on rebuild instead.

Embeddings live in a parallel .npy matrix (one row per memory, same order as JSONL).
"""

import json
import os

from sed import config
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

EMBED_MODEL = config.EMBED_MODEL
FIREWORKS_EMBED_URL = config.FIREWORKS_EMBED_URL


class MemoryStore:
    def __init__(
        self,
        jsonl_path: str = "memories.jsonl",
        embed_path: str = "memories.npy",
        model_name: str = EMBED_MODEL,
        fireworks_api_key: Optional[str] = None,
        content_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.embed_path = Path(embed_path)
        self.model_name = model_name

        self._fw_api_key = fireworks_api_key or os.environ.get("FIREWORKS_API_KEY", "")
        if not self._fw_api_key:
            raise ValueError(
                "Fireworks API key required. Pass fireworks_api_key= or set FIREWORKS_API_KEY."
            )

        # If provided, embedding text is derived from metadata via this function instead of
        # being stored in the "content" field. Enables cleaner JSONL for structured stores (e.g. L2).
        self.content_fn = content_fn

        # in-memory state
        self.memories: List[Dict[str, Any]] = []   # list of {id, timestamp, metadata} (+ "content" if content_fn is None)
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memories and embeddings from disk."""
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                self.memories = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded {len(self.memories)} memories from {self.jsonl_path}")
        else:
            self.memories = []

        if self.embed_path.exists() and self.memories:
            self.embeddings = np.load(str(self.embed_path))
            if self.embeddings.shape[0] != len(self.memories):
                logger.warning("Embedding count mismatch — re-embedding all memories")
                self._rebuild_embeddings()
        elif self.memories:
            logger.warning("Embeddings file missing — re-embedding all memories")
            self._rebuild_embeddings()
        else:
            self.embeddings = None

    def _save(self) -> None:
        """Flush memories and embeddings to disk."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for mem in self.memories:
                f.write(json.dumps(mem, ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(str(self.embed_path), self.embeddings)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts via Fireworks API → (N, D) float32 unit-normalised array."""
        all_vecs: List[List[float]] = []
        batch_size = 32
        headers = {
            "Authorization": f"Bearer {self._fw_api_key}",
            "Content-Type": "application/json",
        }
        retryable = {429, 500, 502, 503, 504}
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(5):
                resp = requests.post(
                    FIREWORKS_EMBED_URL,
                    headers=headers,
                    json={"model": self.model_name, "input": batch},
                    timeout=60,
                )
                if resp.status_code not in retryable or attempt == 4:
                    resp.raise_for_status()
                    break
                wait = 2 ** attempt
                logger.warning(
                    "Embedding API %s (attempt %d/5), retrying in %ds",
                    resp.status_code,
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            all_vecs.extend(item["embedding"] for item in data)

        matrix = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def _get_embed_text(self, entry: Dict[str, Any]) -> str:
        if self.content_fn is not None:
            return self.content_fn(entry)
        return entry["content"]

    def _rebuild_embeddings(self) -> None:
        if not self.memories:
            self.embeddings = None
            return
        texts = [self._get_embed_text(m) for m in self.memories]
        self.embeddings = self._embed(texts)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Add a new memory. Returns the assigned id.

        When content_fn is set, `content` is used only for embedding and is NOT stored
        in the JSONL entry — the embedding text will be reconstructed from metadata on
        rebuild. When content_fn is None, `content` is stored as usual.
        """
        mid = memory_id or str(uuid.uuid4())
        entry: Dict[str, Any] = {
            "id": mid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        if self.content_fn is None:
            entry["content"] = content
        self.memories.append(entry)

        # append embedding row
        embed_text = self.content_fn(entry) if self.content_fn is not None else content
        vec = self._embed([embed_text])        # (1, D)
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

        self._save()
        logger.debug(f"ADD memory [{mid}]: {embed_text[:80]}")
        return mid

    def update(self, memory_id: str, content: str) -> bool:
        """Update the content (and embedding) of an existing memory.

        When content_fn is set, `content` is ignored — the new embedding text is derived
        from the entry's current metadata via content_fn. Pass any string as a no-op placeholder.
        """
        for i, mem in enumerate(self.memories):
            if mem["id"] == memory_id:
                if self.content_fn is None:
                    mem["content"] = content
                mem["timestamp"] = datetime.now(timezone.utc).isoformat()
                # replace embedding row
                embed_text = self.content_fn(mem) if self.content_fn is not None else content
                vec = self._embed([embed_text])
                if self.embeddings is None:
                    self._rebuild_embeddings()
                self.embeddings[i] = vec[0]
                self._save()
                logger.debug(f"UPDATE memory [{memory_id}]: {embed_text[:80]}")
                return True
        logger.warning(f"update: memory {memory_id} not found")
        return False

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by id."""
        for i, mem in enumerate(self.memories):
            if mem["id"] == memory_id:
                self.memories.pop(i)
                if self.embeddings is not None:
                    self.embeddings = np.delete(self.embeddings, i, axis=0)
                    if self.embeddings.shape[0] == 0:
                        self.embeddings = None
                self._save()
                logger.debug(f"DELETE memory [{memory_id}]")
                return True
        return False

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for mem in self.memories:
            if mem["id"] == memory_id:
                return mem
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.memories)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return up to top_k most-similar memories with scores.
        Each result: {id, content, timestamp, metadata, score}
        """
        if not self.memories or self.embeddings is None:
            return []

        q_vec = self._embed([query])          # (1, D)
        sims = (q_vec @ self.embeddings.T).squeeze(0)  # (N,)

        k = min(top_k, len(self.memories))
        top_idx = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_idx:
            mem = dict(self.memories[idx])
            mem["score"] = float(sims[idx])
            results.append(mem)
        return results

    def find_similar(self, text: str, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """
        Return (memory_id, score) of the most similar existing memory if above threshold,
        else None. Used for deduplication before adding.
        """
        results = self.search(text, top_k=1)
        if results and results[0]["score"] >= threshold:
            return results[0]["id"], results[0]["score"]
        return None

    def __len__(self) -> int:
        return len(self.memories)
