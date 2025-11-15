"""Lightweight vector store for retrieval augmented generation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


@dataclass(slots=True)
class DocumentChunk:
    identifier: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class VectorizedDocument:
    chunk: DocumentChunk
    vector: dict[str, float]


class SimpleVectorStore:
    """Very small in-memory vector store."""

    def __init__(self) -> None:
        self._documents: list[VectorizedDocument] = []

    def add_documents(self, chunks: Iterable[DocumentChunk]) -> None:
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            if not tokens:
                continue
            vector = self._normalize(Counter(tokens))
            self._documents.append(VectorizedDocument(chunk=chunk, vector=vector))

    def similarity_search(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        query_vector = self._normalize(Counter(tokens))
        scored = [
            (self._cosine(query_vector, doc.vector), doc.chunk)
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in scored[:top_k] if score > 0]

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _normalize(self, counts: Counter[str]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {token: value / norm for token, value in counts.items()}

    def _cosine(self, first: dict[str, float], second: dict[str, float]) -> float:
        return sum(first.get(token, 0.0) * second.get(token, 0.0) for token in first.keys())


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 60) -> list[str]:
    """Split text into overlapping chunks."""

    sanitized = text.strip()
    if not sanitized:
        return []
    words = sanitized.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            start = end
        else:
            start = max(0, end - overlap)
    return chunks


def load_text_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")
