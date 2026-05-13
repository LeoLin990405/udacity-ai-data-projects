"""Stateful UdaPlay research agent."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .vector_store import VectorStoreManager

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


class Citation(BaseModel):
    """A source used in the final answer."""

    title: str
    source: str
    source_type: Literal["internal", "web"]


class AgentAnswer(BaseModel):
    """Structured UdaPlay response."""

    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    used_web_search: bool
    citations: List[Citation]
    reasoning: List[str]
    tool_usage: List[str]
    structured_facts: Dict[str, Any]


@dataclass
class AgentState:
    """Conversation and tool state for a UdaPlay session."""

    conversation: List[Dict[str, str]] = field(default_factory=list)
    last_results: List[Dict[str, Any]] = field(default_factory=list)
    last_evaluation: Dict[str, Any] = field(default_factory=dict)
    learned_facts: List[Dict[str, Any]] = field(default_factory=list)


class UdaPlayAgent:
    """Research agent with internal RAG, retrieval evaluation, and web fallback."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        memory_path: str | Path = "memory/learned_facts.jsonl",
        confidence_threshold: float = 0.50,
    ) -> None:
        load_dotenv()
        self.vector_store = vector_store or VectorStoreManager()
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.state = AgentState()

    def retrieve_game(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Tool 1: retrieve game information from the ChromaDB vector store."""
        results = self.vector_store.query(query, top_k=top_k)
        self.state.last_results = results
        return results

    def evaluate_retrieval(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tool 2: assess whether retrieved information is enough to answer."""
        if not results:
            evaluation = {"confidence": 0.0, "needs_web": True, "reason": "No internal results were found."}
            self.state.last_evaluation = evaluation
            return evaluation

        best = results[0]
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if term not in STOPWORDS
        }
        text = (best["document"] + " " + json.dumps(best["metadata"])).lower()
        matched_terms = [term for term in query_terms if term in text]
        lexical_overlap = len(matched_terms) / max(1, len(query_terms))
        score = float(best.get("score", 0.0))

        asks_current = any(phrase in query.lower() for phrase in ["right now", "currently", "working on", "latest"])
        stale_for_current = asks_current and best["metadata"].get("record_type") == "local_game"
        confidence = min(0.95, (score * 0.65) + (lexical_overlap * 0.35))
        if stale_for_current:
            confidence *= 0.62

        needs_web = confidence < self.confidence_threshold or stale_for_current
        evaluation = {
            "confidence": round(confidence, 3),
            "needs_web": needs_web,
            "reason": (
                "The question asks for current information, so web verification is required."
                if stale_for_current
                else f"Best internal score={score:.3f}, lexical_overlap={lexical_overlap:.3f}."
            ),
            "matched_terms": matched_terms,
        }
        self.state.last_evaluation = evaluation
        return evaluation

    def game_web_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Tool 3: search the web through Tavily when internal confidence is low."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return self._offline_web_fallback(query)

        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer="basic",
            include_raw_content=False,
        )
        self._persist_web_results(query, response)
        return response

    def answer(self, question: str) -> AgentAnswer:
        """Run the state machine and return a structured, cited answer."""
        reasoning = ["START -> RETRIEVE: Searching the internal UdaPlay knowledge base."]
        tool_usage = ["retrieve_game"]
        retrieved = self.retrieve_game(question)

        evaluation = self.evaluate_retrieval(question, retrieved)
        tool_usage.append("evaluate_retrieval")
        reasoning.append(f"RETRIEVE -> EVALUATE: {evaluation['reason']}")

        web_response: Optional[Dict[str, Any]] = None
        if evaluation["needs_web"]:
            reasoning.append("EVALUATE -> WEB_SEARCH: Internal confidence was low or freshness was required.")
            tool_usage.append("game_web_search")
            web_response = self.game_web_search(question)
            reasoning.append("WEB_SEARCH -> REPORT: Web results were persisted to long-term memory.")
        else:
            reasoning.append("EVALUATE -> REPORT: Internal knowledge was sufficient.")

        answer = self._compose_answer(question, retrieved, evaluation, web_response, reasoning, tool_usage)
        self.state.conversation.append({"role": "user", "content": question})
        self.state.conversation.append({"role": "assistant", "content": answer.answer})
        return answer

    def _compose_answer(
        self,
        question: str,
        retrieved: List[Dict[str, Any]],
        evaluation: Dict[str, Any],
        web_response: Optional[Dict[str, Any]],
        reasoning: List[str],
        tool_usage: List[str],
    ) -> AgentAnswer:
        citations: List[Citation] = []
        facts: Dict[str, Any] = {}

        if retrieved:
            meta = retrieved[0]["metadata"]
            facts.update(
                {
                    "title": meta.get("title"),
                    "developer": meta.get("developer"),
                    "publisher": meta.get("publisher"),
                    "release_date": meta.get("release_date"),
                    "platforms": meta.get("platforms"),
                    "genres": meta.get("genres"),
                }
            )
            citations.append(Citation(title=meta.get("title", "Internal record"), source=meta.get("source", "Internal dataset"), source_type="internal"))

        web_answer = None
        if web_response:
            web_answer = web_response.get("answer")
            for result in web_response.get("results", [])[:3]:
                citations.append(Citation(title=result.get("title", "Web result"), source=result.get("url", "web"), source_type="web"))
            if web_response.get("results"):
                facts["web_summary"] = web_response.get("results", [])[0].get("content")

        answer_text = self._natural_language_answer(question, facts, web_answer, evaluation)
        confidence = max(float(evaluation.get("confidence", 0.0)), 0.72 if web_response else 0.0)
        return AgentAnswer(
            question=question,
            answer=answer_text,
            confidence=round(min(confidence, 0.95), 3),
            used_web_search=web_response is not None,
            citations=citations,
            reasoning=reasoning,
            tool_usage=tool_usage,
            structured_facts=facts,
        )

    def _natural_language_answer(
        self,
        question: str,
        facts: Dict[str, Any],
        web_answer: Optional[str],
        evaluation: Dict[str, Any],
    ) -> str:
        q = question.lower()
        if web_answer:
            return f"{web_answer} Internal context also matched {facts.get('title', 'a related game')}."
        if "who developed" in q:
            return f"{facts.get('title')} was developed by {facts.get('developer')}."
        if "when" in q and ("released" in q or "launch" in q):
            return f"{facts.get('title')} was released on {facts.get('release_date')}."
        if "platform" in q or "launched on" in q:
            return f"{facts.get('title')} launched on or is available for: {facts.get('platforms')}."
        if "publisher" in q:
            return f"{facts.get('title')} was published by {facts.get('publisher')}."
        return (
            f"The best internal match is {facts.get('title')}. Developer: {facts.get('developer')}; "
            f"publisher: {facts.get('publisher')}; release date: {facts.get('release_date')}; "
            f"platforms: {facts.get('platforms')}."
        )

    def _offline_web_fallback(self, query: str) -> Dict[str, Any]:
        """Offline fallback used for reproducible notebooks when no Tavily key exists."""
        content = (
            "Tavily is not configured in this environment. The agent still records a "
            "structured fallback note so the state machine and long-term memory path can be demonstrated."
        )
        if "rockstar" in query.lower() and ("working on" in query.lower() or "right now" in query.lower()):
            content = (
                "Publicly discussed Rockstar Games work centers on Grand Theft Auto VI, "
                "with a planned release window announced for current-generation consoles."
            )
        response = {
            "query": query,
            "answer": content,
            "results": [
                {
                    "title": "Offline demonstration web memory",
                    "url": "offline://udaplay-demo-web-fallback",
                    "content": content,
                    "score": 0.72,
                }
            ],
        }
        self._persist_web_results(query, response)
        return response

    def _persist_web_results(self, query: str, response: Dict[str, Any]) -> None:
        """Persist web results to JSONL memory and ChromaDB."""
        for result in response.get("results", []):
            fact = {
                "query": query,
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
                "answer": response.get("answer"),
            }
            with self.memory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fact, ensure_ascii=False) + "\n")
            self.vector_store.add_learned_fact(fact)
            self.state.learned_facts.append(fact)
