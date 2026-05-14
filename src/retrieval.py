from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedDocument:
    doc_id: str
    content: str
    score: float
    path: str


class DocumentRetriever:
    """Small deterministic retriever for the project documents."""

    def __init__(self, documents_dir: str | Path = "documents"):
        self.documents_dir = Path(documents_dir)
        self.documents: dict[str, str] = {}
        self.paths: dict[str, str] = {}
        self._load_documents()

    def _load_documents(self) -> None:
        self.documents_dir.mkdir(exist_ok=True)
        for path in sorted(self.documents_dir.glob("*.txt")):
            doc_id = path.stem
            self.documents[doc_id] = path.read_text(encoding="utf-8")
            self.paths[doc_id] = str(path)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9$%.]+", text.lower()))

    def search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        query_tokens = self._tokens(query)
        results: list[RetrievedDocument] = []
        for doc_id, content in self.documents.items():
            doc_tokens = self._tokens(doc_id.replace("_", " ") + " " + content)
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            if doc_id.lower() in query.lower():
                score += 1.0
            results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    path=self.paths[doc_id],
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    @staticmethod
    def format_context(documents: list[RetrievedDocument]) -> str:
        parts = []
        for doc in documents:
            parts.append(
                f"--- SOURCE: {doc.doc_id} | score={doc.score:.3f} ---\n{doc.content.strip()}"
            )
        return "\n\n".join(parts)
