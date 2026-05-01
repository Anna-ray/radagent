"""
radagent.rag.retriever
----------------------
Pure-runtime retriever. Loads a prebuilt FAISS index + chunks JSONL,
exposes query(text, k) -> list[Passage].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .passage import Passage

DEFAULT_MODEL = "BAAI/bge-m3"


class RadRetriever:
    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        manifest_path: str,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.manifest_path = Path(manifest_path)

        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        self.embed_model_name = self.manifest.get("embed_model", model_name)
        self.embed_dim = int(self.manifest["embed_dim"])

        self.chunks: list[dict] = []
        with open(self.chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        if len(self.chunks) != int(self.manifest["n_chunks"]):
            raise ValueError(
                f"chunks.jsonl has {len(self.chunks)} entries but manifest "
                f"says {self.manifest['n_chunks']}"
            )

        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"FAISS index has {self.index.ntotal} vectors but "
                f"{len(self.chunks)} chunks present"
            )

        self._embedder = None
        self._device = device

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[rag] loading embedder {self.embed_model_name} ...", flush=True)
            self._embedder = SentenceTransformer(
                self.embed_model_name, device=self._device
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_embedder()
        emb = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def query(
        self,
        text: str,
        k: int = 3,
        finding_filter: Optional[list[str]] = None,
    ) -> list[Passage]:
        if not text.strip():
            return []
        q_emb = self._embed([text])
        oversample = k * 5 if finding_filter else k
        oversample = min(oversample, len(self.chunks))
        scores, idxs = self.index.search(q_emb, oversample)
        scores, idxs = scores[0], idxs[0]

        out: list[Passage] = []
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            ch = self.chunks[i]
            if finding_filter:
                keys = set(ch.get("finding_keys", []))
                if not keys.intersection(finding_filter):
                    continue
            out.append(Passage(
                text=ch["text"],
                source_url=ch["source_url"],
                title=ch["title"],
                section=ch["section"],
                chunk_id=ch["chunk_id"],
                finding_keys=list(ch.get("finding_keys", [])),
                score=float(s),
                license=ch.get("license", "CC-BY-NC-SA"),
                source=ch.get("source", "radiopaedia"),
            ))
            if len(out) >= k:
                break
        return out
