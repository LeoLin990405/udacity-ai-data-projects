from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .schemas import AnswerResponse, CalculationResponse, MemoryUpdate, SummaryResponse, UserIntent


class _StructuredFake:
    def __init__(self, schema: type):
        self.schema = schema

    @staticmethod
    def _text(prompt: Any) -> str:
        return str(prompt)

    def invoke(self, prompt: Any):
        text = self._text(prompt)
        lower = text.lower()
        if self.schema is UserIntent:
            request_lower = _last_user_request(text).lower() or lower
            if re.search(r"\b(calculate|sum|total|difference|percent|ratio|multiply)\b", request_lower):
                intent = "calculation"
            elif re.search(r"\b(summarize|summary|key points|overview|brief)\b", request_lower):
                intent = "summarization"
            elif request_lower.strip():
                intent = "qa"
            else:
                intent = "unknown"
            return UserIntent(intent_type=intent, confidence=0.95, reasoning=f"Keyword and request pattern indicate {intent}.")

        sources = re.findall(r"SOURCE:\s*([A-Za-z0-9_-]+)", text)
        sources = list(dict.fromkeys(sources)) or ["unknown"]

        if self.schema is AnswerResponse:
            answer = "The retrieved documents indicate that " + _first_fact(text)
            return AnswerResponse(
                question=_last_user_request(text),
                answer=answer,
                sources=sources,
                confidence=0.84,
                timestamp=datetime.now(timezone.utc),
            )

        if self.schema is SummaryResponse:
            key_points = _key_points(text)
            return SummaryResponse(
                summary=" ".join(key_points[:3]) or "The document contains operational and financial details.",
                key_points=key_points[:5],
                sources=sources,
                confidence=0.86,
                timestamp=datetime.now(timezone.utc),
            )

        if self.schema is CalculationResponse:
            expression_match = re.search(r"CALCULATOR_EXPRESSION:\s*(.+)", text)
            result_match = re.search(r"CALCULATOR_RESULT:\s*(.+)", text)
            expression = expression_match.group(1).strip() if expression_match else "0"
            result = result_match.group(1).strip() if result_match else "0"
            return CalculationResponse(
                question=_last_user_request(text),
                expression=expression,
                result=result,
                explanation=f"The calculator evaluated `{expression}` and returned `{result}`.",
                sources=sources,
                confidence=0.88,
                timestamp=datetime.now(timezone.utc),
            )

        if self.schema is MemoryUpdate:
            active_match = re.search(r"Active documents:\s*(\[.*?\])", text)
            active_docs = sources
            if active_match:
                active_docs = re.findall(r"'([^']+)'|\"([^\"]+)\"", active_match.group(1))
                active_docs = [left or right for left, right in active_docs]
            return MemoryUpdate(
                conversation_summary="The user discussed retrieved financial and healthcare documents.",
                active_documents=active_docs or sources,
            )

        return self.schema()


class FakeDocumentLLM:
    """Small test double with the LangChain structured-output surface."""

    def with_structured_output(self, schema: type):
        return _StructuredFake(schema)

    def invoke(self, prompt: Any):
        return "FakeDocumentLLM response"


def _last_user_request(text: str) -> str:
    if "Current user input:" in text:
        return text.split("Current user input:")[-1].strip().splitlines()[0].strip()
    if "User request:" in text:
        return text.split("User request:")[-1].strip().splitlines()[0].strip()
    match = re.findall(r"User request:\s*(.+)", text)
    if match:
        return match[-1].strip()
    match = re.findall(r"Current user input:\s*(.+)", text)
    return match[-1].strip() if match else ""


def _first_fact(text: str) -> str:
    context = text.split("Retrieved context:")[-1] if "Retrieved context:" in text else text
    for line in context.splitlines():
        clean = line.strip()
        if any(keyword in clean.lower() for keyword in ["revenue was", "operating expenses", "net income", "readmission", "patient satisfaction", "average patient"]):
            return clean[:240]
    for line in text.splitlines():
        line = line.strip()
        skipped_prefixes = (
            "---",
            "System:",
            "Human:",
            "human:",
            "ai:",
            "Conversation history",
            "Retrieved context",
            "User request",
            "Answer only",
            "If the context",
            "Create concise",
            "include source",
            "Do not",
        )
        if line and not line.startswith(skipped_prefixes):
            if len(line) > 20:
                return line[:240]
    return "the requested information is present in the retrieved context."


def _key_points(text: str) -> list[str]:
    points = []
    context = text.split("Retrieved context:")[-1] if "Retrieved context:" in text else text
    for raw in context.splitlines():
        line = raw.strip("- ").strip()
        skipped_prefixes = (
            "---",
            "System:",
            "Human:",
            "human:",
            "ai:",
            "Conversation history",
            "Retrieved context",
            "User request",
            "Create concise",
            "include source",
            "Do not",
        )
        if len(line) > 25 and not line.startswith(skipped_prefixes):
            points.append(line[:220])
    return points or ["Retrieved context was reviewed.", "The answer should cite the active source documents."]
