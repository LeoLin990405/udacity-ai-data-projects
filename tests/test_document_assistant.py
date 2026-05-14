from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.assistant import AssistantSession, DocumentAssistant
from src.schemas import AnswerResponse, UserIntent
from src.tools import ToolLogger, create_calculator_tool


def test_schemas_validate_bounds_and_intent_values():
    response = AnswerResponse(question="q", answer="a", confidence=0.75)
    assert response.confidence == 0.75
    with pytest.raises(ValidationError):
        AnswerResponse(question="q", answer="a", confidence=1.5)
    with pytest.raises(ValidationError):
        UserIntent(intent_type="joke", confidence=0.5, reasoning="invalid")


def test_calculator_tool_logs_and_handles_errors(tmp_path):
    logger = ToolLogger(tmp_path, "calc-session")
    calculator = create_calculator_tool(logger)
    assert calculator.invoke("2 + 3 * 4") == "14"
    assert calculator.invoke("__import__('os').system('echo unsafe')").startswith("Error:")
    log_path = tmp_path / "calc-session_tool_calls.jsonl"
    assert log_path.exists()
    assert "calculator" in log_path.read_text(encoding="utf-8")


def test_end_to_end_routes_all_intents_and_persists(tmp_path):
    root = Path(__file__).resolve().parents[1]
    session = AssistantSession(session_id="pytest-session", user_id="pytest-user")
    assistant = DocumentAssistant(
        documents_dir=root / "documents",
        logs_dir=tmp_path / "logs",
        sessions_dir=tmp_path / "sessions",
        session=session,
    )

    qa = assistant.process_message("What was Q2 revenue?")
    assert qa["intent"].intent_type == "qa"
    assert "qa_agent" in qa["actions_taken"]
    assert qa["current_response"]["sources"]

    summary = assistant.process_message("Summarize the healthcare quality report.")
    assert summary["intent"].intent_type == "summarization"
    assert "summarization_agent" in summary["actions_taken"]
    assert "key_points" in summary["current_response"]

    calc = assistant.process_message("Calculate the difference between 1410000 and 1250000.")
    assert calc["intent"].intent_type == "calculation"
    assert "calculation_agent" in calc["actions_taken"]
    assert calc["current_response"]["result"] == "160000.0"
    assert "update_memory" in calc["actions_taken"]

    assert (tmp_path / "sessions" / "pytest-session.jsonl").exists()
    assert (tmp_path / "logs" / "pytest-session_tool_calls.jsonl").exists()
