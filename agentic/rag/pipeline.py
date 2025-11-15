"""RAG utilities specific to the healthcare agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .document_store import DocumentChunk, SimpleVectorStore, chunk_text, load_text_file


@dataclass(slots=True)
class RetrievedContext:
    """Container returned to the agents after retrieval."""

    question: str
    passages: list[DocumentChunk] = field(default_factory=list)

    def as_bullet_list(self) -> str:
        bullets = [
            f"- ({chunk.metadata.get('source', chunk.identifier)}) {chunk.text.strip()}"
            for chunk in self.passages
        ]
        return "\n".join(bullets)


class HealthcareRAGPipeline:
    """Small RAG helper built around the SimpleVectorStore."""

    def __init__(
        self,
        knowledge_base_path: str | Path,
        chunk_size: int = 450,
        chunk_overlap: int = 60,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = SimpleVectorStore()
        self.knowledge_base_path = Path(knowledge_base_path)
        if self.knowledge_base_path.exists():
            self.ingest_corpus([self.knowledge_base_path])

    def ingest_corpus(self, files: Iterable[str | Path]) -> None:
        chunks: list[DocumentChunk] = []
        for index, file_path in enumerate(files):
            content = load_text_file(file_path)
            if not content:
                continue
            for chunk_index, chunk_text_value in enumerate(
                chunk_text(content, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            ):
                chunks.append(
                    DocumentChunk(
                        identifier=f"{index}-{chunk_index}",
                        text=chunk_text_value,
                        metadata={"source": str(file_path)},
                    )
                )
        self.vector_store.add_documents(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> RetrievedContext:
        passages = self.vector_store.similarity_search(question, top_k=top_k)
        return RetrievedContext(question=question, passages=passages)

    def describe(self) -> dict[str, str | int]:
        return {
            "knowledge_base": str(self.knowledge_base_path),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
