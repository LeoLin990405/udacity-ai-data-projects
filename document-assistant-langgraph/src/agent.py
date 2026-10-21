import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .prompts import get_chat_prompt_template, get_intent_classification_prompt
from .schemas import AnswerResponse, CalculationResponse, MemoryUpdate, SummaryResponse, UserIntent


class AgentState(TypedDict, total=False):
    user_input: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: UserIntent
    next_step: str
    conversation_summary: str
    active_documents: list[str]
    current_response: dict[str, Any]
    tools_used: list[str]
    session_id: str
    user_id: str
    actions_taken: Annotated[list[str], operator.add]


def _config_value(config: dict | None, key: str, default: Any = None) -> Any:
    return (config or {}).get("configurable", {}).get(key, default)


def _history_text(state: AgentState) -> str:
    parts = []
    for message in state.get("messages", [])[-8:]:
        role = getattr(message, "type", "message")
        content = getattr(message, "content", str(message))
        parts.append(f"{role}: {content}")
    return "\n".join(parts) or "(no prior conversation)"


def _doc_sources(context: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"SOURCE:\s*([A-Za-z0-9_-]+)", context)))


def _tool_by_name(config: dict | None, name: str):
    tools = _config_value(config, "tools", [])
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    raise KeyError(f"Tool not configured: {name}")


def _invoke_tool(tool, value: str) -> str:
    try:
        return tool.invoke(value)
    except TypeError:
        return tool.run(value)


def classify_intent(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Classify the user request and decide which graph node should run next."""

    llm = _config_value(config, "llm")
    structured_llm = llm.with_structured_output(UserIntent)
    prompt = get_intent_classification_prompt().format(
        user_input=state["user_input"],
        conversation_history=_history_text(state),
    )
    intent = structured_llm.invoke(prompt)
    route_map = {
        "qa": "qa_agent",
        "summarization": "summarization_agent",
        "calculation": "calculation_agent",
    }
    next_step = route_map.get(intent.intent_type, "qa_agent")
    return {
        "intent": intent,
        "next_step": next_step,
        "actions_taken": ["classify_intent"],
    }


def qa_agent(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Answer a fact-seeking question from retrieved document context."""

    llm = _config_value(config, "llm")
    document_reader = _tool_by_name(config, "document_reader")
    context = _invoke_tool(document_reader, state["user_input"])
    prompt = get_chat_prompt_template("qa").format(
        conversation_history=_history_text(state),
        context=context,
        user_input=state["user_input"],
    )
    response = llm.with_structured_output(AnswerResponse).invoke(prompt)
    return {
        "current_response": response.model_dump(mode="json"),
        "messages": [AIMessage(content=response.answer)],
        "active_documents": response.sources or _doc_sources(context),
        "tools_used": ["document_reader"],
        "next_step": "update_memory",
        "actions_taken": ["qa_agent"],
    }


def summarization_agent(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Summarize retrieved documents and extract key points."""

    llm = _config_value(config, "llm")
    document_reader = _tool_by_name(config, "document_reader")
    context = _invoke_tool(document_reader, state["user_input"])
    prompt = get_chat_prompt_template("summarization").format(
        conversation_history=_history_text(state),
        context=context,
        user_input=state["user_input"],
    )
    response = llm.with_structured_output(SummaryResponse).invoke(prompt)
    return {
        "current_response": response.model_dump(mode="json"),
        "messages": [AIMessage(content=response.summary)],
        "active_documents": response.sources or _doc_sources(context),
        "tools_used": ["document_reader"],
        "next_step": "update_memory",
        "actions_taken": ["summarization_agent"],
    }


def _infer_expression(text: str) -> str:
    numbers = [float(value.replace(",", "")) for value in re.findall(r"(?<![A-Za-z])\$?([0-9][0-9,]*(?:\.[0-9]+)?)", text)]
    lower = text.lower()
    if not numbers:
        return "0"
    if any(word in lower for word in ["difference", "subtract", "less", "decrease"]):
        return " - ".join(str(n) for n in numbers[:2])
    if any(word in lower for word in ["average", "mean"]):
        return f"({' + '.join(str(n) for n in numbers)}) / {len(numbers)}"
    if any(word in lower for word in ["multiply", "product"]):
        return " * ".join(str(n) for n in numbers[:2])
    if any(word in lower for word in ["ratio", "divide"]):
        return " / ".join(str(n) for n in numbers[:2])
    return " + ".join(str(n) for n in numbers[:6])


def calculation_agent(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Retrieve document data, use the calculator tool, and return a structured calculation."""

    llm = _config_value(config, "llm")
    document_reader = _tool_by_name(config, "document_reader")
    calculator = _tool_by_name(config, "calculator")
    context = _invoke_tool(document_reader, state["user_input"])
    expression = _infer_expression(state["user_input"] + "\n" + context)
    calculator_result = _invoke_tool(calculator, expression)
    prompt = (
        get_chat_prompt_template("calculation").format(
            conversation_history=_history_text(state),
            context=context,
            user_input=state["user_input"],
        )
        + f"\n\nCALCULATOR_EXPRESSION: {expression}\nCALCULATOR_RESULT: {calculator_result}"
    )
    response = llm.with_structured_output(CalculationResponse).invoke(prompt)
    return {
        "current_response": response.model_dump(mode="json"),
        "messages": [AIMessage(content=response.explanation)],
        "active_documents": response.sources or _doc_sources(context),
        "tools_used": ["document_reader", "calculator"],
        "next_step": "update_memory",
        "actions_taken": ["calculation_agent"],
    }


def update_memory(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Summarize the turn and persist active document references."""

    llm = _config_value(config, "llm")
    structured_llm = llm.with_structured_output(MemoryUpdate)
    prompt = (
        "Update memory for this document assistant session.\n"
        f"Conversation summary so far: {state.get('conversation_summary', '')}\n"
        f"Recent history:\n{_history_text(state)}\n"
        f"Current response:\n{state.get('current_response', {})}\n"
        f"Active documents: {state.get('active_documents', [])}"
    )
    memory = structured_llm.invoke(prompt)
    return {
        "conversation_summary": memory.conversation_summary,
        "active_documents": memory.active_documents or state.get("active_documents", []),
        "next_step": END,
        "actions_taken": ["update_memory"],
    }


def _route_from_state(state: AgentState) -> str:
    return state.get("next_step", "qa_agent")


def create_workflow():
    """Create and compile the LangGraph workflow with persistent memory."""

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        _route_from_state,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
        },
    )
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)
    return workflow.compile(checkpointer=InMemorySaver())
