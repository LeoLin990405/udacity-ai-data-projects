from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .agent import create_workflow
from .fake_llm import FakeDocumentLLM
from .retrieval import DocumentRetriever
from .tools import ToolLogger, create_calculator_tool, create_document_reader_tool


@dataclass
class AssistantSession:
    session_id: str = field(default_factory=lambda: f"session-{uuid4().hex[:8]}")
    user_id: str = "local-user"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentAssistant:
    """Main assistant wrapper that configures tools, LLM, graph, and session persistence."""

    def __init__(
        self,
        documents_dir: str | Path = "documents",
        logs_dir: str | Path = "logs",
        sessions_dir: str | Path = "sessions",
        session: AssistantSession | None = None,
        llm=None,
    ):
        load_dotenv()
        self.session = session or AssistantSession()
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.retriever = DocumentRetriever(documents_dir)
        self.tool_logger = ToolLogger(logs_dir, self.session.session_id)
        self.tools = [
            create_document_reader_tool(self.retriever, self.tool_logger),
            create_calculator_tool(self.tool_logger),
        ]
        self.llm = llm or self._build_llm()
        self.workflow = create_workflow()

    @staticmethod
    def _build_llm():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return FakeDocumentLLM()
        try:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": api_key,
                "temperature": 0,
            }
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)
        except Exception:
            return FakeDocumentLLM()

    def process_message(self, user_input: str) -> dict:
        """Process one user message through the graph with persistent memory."""

        config = {
            "configurable": {
                "thread_id": self.session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }
        initial_state = {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "session_id": self.session.session_id,
            "user_id": self.session.user_id,
            "actions_taken": [],
        }
        result = self.workflow.invoke(initial_state, config=config)
        self._save_session_record(user_input, result)
        return result

    def _save_session_record(self, user_input: str, result: dict) -> None:
        path = self.sessions_dir / f"{self.session.session_id}.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session.session_id,
            "user_input": user_input,
            "intent": result.get("intent").model_dump(mode="json") if hasattr(result.get("intent"), "model_dump") else result.get("intent"),
            "current_response": result.get("current_response"),
            "active_documents": result.get("active_documents", []),
            "tools_used": result.get("tools_used", []),
            "actions_taken": result.get("actions_taken", []),
            "conversation_summary": result.get("conversation_summary", ""),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
