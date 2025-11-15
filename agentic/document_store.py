"""Backward-compatible import shims for the lightweight RAG document store."""
from __future__ import annotations

from .rag.document_store import (
    DocumentChunk,
    SimpleVectorStore,
    chunk_text,
    load_text_file,
)

__all__ = [
    "DocumentChunk",
    "SimpleVectorStore",
    "chunk_text",
    "load_text_file",
]
