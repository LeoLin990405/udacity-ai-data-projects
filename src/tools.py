from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ToolLogger:
    """Append JSONL records for tool calls so reviewers can inspect usage history."""

    def __init__(self, logs_dir: str | Path = "logs", session_id: str = "default"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.session_id = session_id

    @property
    def path(self) -> Path:
        return self.logs_dir / f"{self.session_id}_tool_calls.jsonl"

    def log(self, tool_name: str, input_value: Any, output_value: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "tool_name": tool_name,
            "input": input_value,
            "output": output_value,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def _validate_expression(expression: str) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.FloorDiv,
    )
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsafe expression component: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")


def create_calculator_tool(logger: ToolLogger | None = None):
    """Create a safe calculator LangChain tool."""

    logger = logger or ToolLogger()

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic mathematical expression.

        Args:
            expression: A math expression using numbers and +, -, *, /, %, **, or parentheses.
        """

        try:
            _validate_expression(expression)
            value = eval(compile(ast.parse(expression, mode="eval"), "<calculator>", "eval"), {"__builtins__": {}}, {})
            result = str(value)
        except Exception as exc:  # noqa: BLE001 - tool should return graceful errors
            result = f"Error: {exc}"
        logger.log("calculator", expression, result)
        return result

    return calculator


def create_document_reader_tool(retriever, logger: ToolLogger | None = None):
    """Create a tool that retrieves document snippets."""

    logger = logger or ToolLogger()

    @tool
    def document_reader(query: str) -> str:
        """Retrieve relevant document content for a question or task.

        Args:
            query: Search query describing the needed financial or healthcare document content.
        """

        docs = retriever.search(query, k=3)
        result = retriever.format_context(docs)
        logger.log("document_reader", query, [doc.doc_id for doc in docs])
        return result

    return document_reader
