"""
radagent.rag.passage
--------------------
Passage dataclass returned by the retriever.

Attribution is mandatory (Radiopaedia CC-BY-NC-SA): every Passage
carries its source_url so downstream report generation can cite it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Passage:
    text: str
    source_url: str
    title: str
    section: str
    chunk_id: str
    finding_keys: list[str] = field(default_factory=list)
    score: float = 0.0
    license: str = "CC-BY-NC-SA"
    source: str = "radiopaedia"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
