"""Vector store and embedding utilities for UdaPlay."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from dotenv import load_dotenv

STOPWORDS = {
    "a",
    "an",
    "and",
    "is",
    "of",
    "on",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "was",
    "were",
}


@dataclass
class GameDocument:
    """A flattened game record ready for vector search."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]


class EmbeddingProvider:
    """OpenAI embedding provider with deterministic local fallback."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 384) -> None:
        load_dotenv()
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", model)
        self.dimensions = dimensions
        self.use_local = os.getenv("UDAPLAY_USE_LOCAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None

        if self.api_key and not self.use_local:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI when configured, otherwise hash locally."""
        if self._client is not None:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]
        return [self._local_embedding(text) for text in texts]

    def _local_embedding(self, text: str) -> List[float]:
        """Deterministic bag-of-hashes embedding for offline tests and demos."""
        vector = [0.0] * self.dimensions
        for token in text.lower().replace("/", " ").replace("-", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


class VectorStoreManager:
    """Load game JSON data, persist it in ChromaDB, and query it."""

    def __init__(
        self,
        data_dir: str | Path = "data/games",
        chroma_dir: str | Path = "chroma_db",
        collection_name: str = "udaplay_games",
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "UdaPlay local video game knowledge base"},
        )

    def load_game_documents(self) -> List[GameDocument]:
        """Load all local JSON records and convert them to searchable text."""
        documents: List[GameDocument] = []
        for path in sorted(self.data_dir.glob("*.json")):
            records = json.loads(path.read_text(encoding="utf-8"))
            for index, record in enumerate(records):
                title = record["title"]
                aliases = ", ".join(record.get("aliases", []))
                platforms = ", ".join(record.get("platforms", []))
                genres = ", ".join(record.get("genres", []))
                text = (
                    f"Title: {title}\n"
                    f"Aliases: {aliases}\n"
                    f"Developer: {record.get('developer', 'Unknown')}\n"
                    f"Publisher: {record.get('publisher', 'Unknown')}\n"
                    f"Release date: {record.get('release_date', 'Unknown')}\n"
                    f"Platforms: {platforms}\n"
                    f"Genres: {genres}\n"
                    f"Description: {record.get('description', '')}"
                )
                safe_title = "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
                metadata = {
                    "title": title,
                    "developer": record.get("developer", "Unknown"),
                    "publisher": record.get("publisher", "Unknown"),
                    "release_date": record.get("release_date", "Unknown"),
                    "platforms": platforms,
                    "genres": genres,
                    "source": record.get("source", f"Local JSON: {path.name}"),
                    "source_file": path.name,
                    "record_type": "local_game",
                }
                documents.append(GameDocument(f"{path.stem}-{index}-{safe_title}", text, metadata))
        return documents

    def build_index(self, reset: bool = False) -> int:
        """Embed and persist the local game documents in ChromaDB."""
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

        documents = self.load_game_documents()
        if not documents:
            raise ValueError(f"No game JSON files found in {self.data_dir}")

        embeddings = self.embedding_provider.embed_texts([doc.text for doc in documents])
        self.collection.upsert(
            ids=[doc.doc_id for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
        )
        return len(documents)

    def query(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Run semantic search against the ChromaDB collection."""
        query_embedding = self.embedding_provider.embed_texts([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, 8),
            include=["documents", "metadatas", "distances"],
        )
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question.lower())
            if term not in STOPWORDS
        }
        rows: List[Dict[str, Any]] = []
        for index, document in enumerate(results.get("documents", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][index]
            distance = results.get("distances", [[]])[0][index]
            searchable = f"{document} {json.dumps(metadata)}".lower()
            overlap = len([term for term in query_terms if term in searchable]) / max(1, len(query_terms))
            vector_score = 1.0 / (1.0 + float(distance))
            score = (0.55 * vector_score) + (0.45 * overlap)
            rows.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "score": score,
                    "lexical_overlap": overlap,
                }
            )
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:top_k]

    def add_learned_fact(self, fact: Dict[str, Any]) -> str:
        """Persist a web-discovered fact into ChromaDB for future retrieval."""
        title = fact.get("title") or fact.get("query") or "learned-fact"
        content = fact.get("content") or fact.get("answer") or json.dumps(fact)
        source_url = fact.get("url", "web search")
        doc_id = "web-" + hashlib.sha1(f"{title}-{source_url}".encode("utf-8")).hexdigest()[:16]
        text = f"Title: {title}\nSource: {source_url}\nContent: {content}"
        embedding = self.embedding_provider.embed_texts([text])[0]
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[
                {
                    "title": title,
                    "developer": fact.get("developer", "Unknown"),
                    "publisher": fact.get("publisher", "Unknown"),
                    "release_date": fact.get("release_date", "Unknown"),
                    "platforms": fact.get("platforms", "Unknown"),
                    "genres": fact.get("genres", "Unknown"),
                    "source": source_url,
                    "source_file": "memory/learned_facts.jsonl",
                    "record_type": "web_learned_fact",
                }
            ],
        )
        return doc_id


def format_results(results: Iterable[Dict[str, Any]]) -> str:
    """Format retrieval results for reports and notebooks."""
    lines = []
    for position, result in enumerate(results, start=1):
        meta = result["metadata"]
        lines.append(
            f"{position}. {meta.get('title')} | score={result['score']:.3f} | "
            f"source={meta.get('source')}\n{result['document']}"
        )
    return "\n\n".join(lines)
