from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Structured response returned by the Q&A agent."""

    question: str = Field(description="The original user question.")
    answer: str = Field(description="The generated answer.")
    sources: list[str] = Field(default_factory=list, description="Source document IDs used.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence from 0 to 1.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the answer was generated.",
    )


class UserIntent(BaseModel):
    """Structured output for intent classification."""

    intent_type: Literal["qa", "summarization", "calculation", "unknown"] = Field(
        description="The classified user intent."
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why this intent was selected.")


class SummaryResponse(BaseModel):
    """Structured response returned by the summarization agent."""

    summary: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalculationResponse(BaseModel):
    """Structured response returned by the calculation agent."""

    question: str
    expression: str
    result: str
    explanation: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryUpdate(BaseModel):
    """Structured memory update for each turn."""

    conversation_summary: str = Field(default="")
    active_documents: list[str] = Field(default_factory=list)
