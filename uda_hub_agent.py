from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None
    InMemorySaver = None

Route = Literal["resolve", "escalate"]


class TicketState(TypedDict, total=False):
    ticket: Dict[str, Any]
    normalized_text: str
    category: str
    urgency: str
    confidence: float
    route: Route
    knowledge: List[str]
    memory_matches: List[str]
    tools_used: List[str]
    response: str
    escalation_summary: str
    rationale: str
    citations: List[str]
    actions_taken: List[str]


@dataclass
class MemoryStore:
    """Simple long-term memory keyed by customer id and category."""

    records: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, customer_id: str, category: str, note: str) -> None:
        key = f"{customer_id}:{category}"
        self.records.setdefault(key, []).append(note)

    def search(self, customer_id: str, category: str) -> List[str]:
        return list(self.records.get(f"{customer_id}:{category}", []))


@dataclass
class KnowledgeBase:
    path: Path
    chunks: List[str] = field(init=False)

    def __post_init__(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        self.chunks = [chunk.strip() for chunk in text.split("## ") if chunk.strip()]

    def search(self, query: str, limit: int = 2) -> List[str]:
        terms = {t.strip(".,!?;:()[]").lower() for t in query.split() if len(t) > 2}
        scored = []
        for chunk in self.chunks:
            lower = chunk.lower()
            score = sum(1 for term in terms if term in lower)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]


class UDAHubAgent:
    """Autonomous support agent with LangGraph orchestration and deterministic tools."""

    def __init__(self, knowledge_path: str = "data/faqs.md") -> None:
        self.knowledge_base = KnowledgeBase(Path(knowledge_path))
        self.memory = MemoryStore()
        self.graph = self._build_graph()

    def invoke(self, ticket: Dict[str, Any], thread_id: str = "default") -> TicketState:
        initial: TicketState = {"ticket": ticket, "tools_used": [], "actions_taken": []}
        if self.graph is not None:
            config = {"configurable": {"thread_id": thread_id}}
            return self.graph.invoke(initial, config=config)
        return self._fallback_invoke(initial)

    def _build_graph(self):
        if StateGraph is None:
            return None

        graph = StateGraph(TicketState)
        graph.add_node("supervisor", self.supervisor_agent)
        graph.add_node("classifier", self.classifier_agent)
        graph.add_node("knowledge", self.knowledge_agent)
        graph.add_node("resolver", self.resolver_agent)
        graph.add_node("escalation", self.escalation_agent)
        graph.add_node("memory", self.memory_agent)
        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "classifier")
        graph.add_edge("classifier", "knowledge")
        graph.add_conditional_edges(
            "knowledge",
            self.route_decision,
            {"resolve": "resolver", "escalate": "escalation"},
        )
        graph.add_edge("resolver", "memory")
        graph.add_edge("escalation", "memory")
        graph.add_edge("memory", END)
        checkpointer = InMemorySaver() if InMemorySaver else None
        return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()

    def _fallback_invoke(self, state: TicketState) -> TicketState:
        for fn in [self.supervisor_agent, self.classifier_agent, self.knowledge_agent]:
            state.update(fn(state))
        branch = self.resolver_agent if self.route_decision(state) == "resolve" else self.escalation_agent
        state.update(branch(state))
        state.update(self.memory_agent(state))
        return state

    def supervisor_agent(self, state: TicketState) -> TicketState:
        ticket = state["ticket"]
        text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".strip()
        return {"normalized_text": text, "actions_taken": state.get("actions_taken", []) + ["supervisor"]}

    def classifier_agent(self, state: TicketState) -> TicketState:
        text = state["normalized_text"].lower()
        metadata = state["ticket"]
        category = "general"
        if any(term in text for term in ["refund", "chargeback", "money back"]):
            category = "refund"
        elif any(term in text for term in ["password", "login", "mfa", "account access"]):
            category = "account_access"
        elif any(term in text for term in ["invoice", "billing", "tax"]):
            category = "billing"
        elif any(term in text for term in ["outage", "down", "error", "production"]):
            category = "technical_outage"
        elif "cancel" in text:
            category = "cancellation"

        urgency = str(metadata.get("urgency", "normal")).lower()
        high_risk = urgency in {"high", "urgent", "critical"} or category in {"technical_outage", "account_access"}
        confidence = 0.9 if category != "general" else 0.55
        route: Route = "escalate" if high_risk or confidence < 0.65 else "resolve"
        rationale = f"Classified as {category} with {urgency} urgency; route={route}."
        return {
            "category": category,
            "urgency": urgency,
            "confidence": confidence,
            "route": route,
            "rationale": rationale,
            "actions_taken": state.get("actions_taken", []) + ["classifier"],
        }

    def knowledge_agent(self, state: TicketState) -> TicketState:
        query = f"{state['category']} {state['normalized_text']}"
        knowledge = self.knowledge_base.search(query)
        ticket = state["ticket"]
        memory = self.memory.search(str(ticket.get("customer_id", "anonymous")), state["category"])
        return {
            "knowledge": knowledge,
            "memory_matches": memory,
            "tools_used": state.get("tools_used", []) + ["search_knowledge_base", "recall_long_term_memory"],
            "citations": [f"KB:{chunk.splitlines()[0]}" for chunk in knowledge],
            "actions_taken": state.get("actions_taken", []) + ["knowledge"],
        }

    def route_decision(self, state: TicketState) -> Route:
        return state.get("route", "escalate")

    def resolver_agent(self, state: TicketState) -> TicketState:
        category = state["category"]
        tools = list(state.get("tools_used", []))
        if category == "refund":
            tools.append("refund_policy_check")
            response = "Your request appears eligible for standard refund review. I verified the policy window and will start the refund workflow for the order on file."
        elif category == "billing":
            tools.append("invoice_lookup")
            response = "I found the billing path for this request. We can resend the invoice to the billing email and route tax corrections to billing support if needed."
        elif category == "cancellation":
            tools.append("cancellation_workflow")
            response = "I can help begin cancellation. Please confirm the account id and acknowledge the data retention policy before final closure."
        else:
            response = "I found relevant support guidance and prepared a standard response."
        return {"response": response, "tools_used": tools, "actions_taken": state.get("actions_taken", []) + ["resolver"]}

    def escalation_agent(self, state: TicketState) -> TicketState:
        ticket = state["ticket"]
        summary = (
            f"Escalate {state['category']} ticket for customer {ticket.get('customer_id', 'unknown')}. "
            f"Urgency={state['urgency']}. Reason: {state['rationale']} "
            f"Customer text: {state['normalized_text']}"
        )
        response = "I am escalating this to a specialist with the relevant context so it can be handled safely and quickly."
        return {
            "escalation_summary": summary,
            "response": response,
            "tools_used": state.get("tools_used", []) + ["create_escalation_summary"],
            "actions_taken": state.get("actions_taken", []) + ["escalation"],
        }

    def memory_agent(self, state: TicketState) -> TicketState:
        ticket = state["ticket"]
        note = f"{state['route']}:{state['category']}:{state['response']}"
        self.memory.add(str(ticket.get("customer_id", "anonymous")), state["category"], note)
        return {"actions_taken": state.get("actions_taken", []) + ["memory"]}


def demo_tickets() -> List[Dict[str, Any]]:
    return [
        {"customer_id": "C-100", "platform": "Zendesk", "urgency": "normal", "subject": "Refund request", "body": "I bought the annual plan last week and need a refund."},
        {"customer_id": "C-200", "platform": "Intercom", "urgency": "critical", "subject": "Production outage", "body": "Our workspace is down and API calls return 503 errors."},
        {"customer_id": "C-300", "platform": "Freshdesk", "urgency": "normal", "subject": "Need invoice", "body": "Please resend the April invoice to our billing email."},
    ]
