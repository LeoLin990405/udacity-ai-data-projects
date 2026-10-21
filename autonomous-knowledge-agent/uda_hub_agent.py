from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from database import DB_PATH, SupportTools, initialize_database

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
    ticket_id: str
    normalized_text: str
    category: str
    urgency: str
    confidence: float
    knowledge_confidence: float
    route: Route
    knowledge: List[Dict[str, Any]]
    memory_matches: List[Dict[str, Any]]
    tools_used: List[str]
    response: str
    escalation_summary: str
    rationale: str
    citations: List[str]
    actions_taken: List[str]


class UDAHubAgent:
    """Autonomous support agent with LangGraph orchestration and database-backed tools."""

    def __init__(self, db_path: str | Path = DB_PATH, initialize: bool = True) -> None:
        self.db_path = Path(db_path)
        if initialize:
            initialize_database(self.db_path)
        self.tools = SupportTools(self.db_path)
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
        ticket_id = self.tools.persist_ticket_and_message(metadata, category)
        return {
            "ticket_id": ticket_id,
            "category": category,
            "urgency": urgency,
            "confidence": confidence,
            "route": route,
            "rationale": rationale,
            "actions_taken": state.get("actions_taken", []) + ["classifier"],
        }

    def knowledge_agent(self, state: TicketState) -> TicketState:
        query = f"{state['category']} {state['normalized_text']}"
        ticket = state["ticket"]
        ticket_id = state.get("ticket_id")
        knowledge_result = self.tools.search_knowledge(query, state["category"], ticket_id=ticket_id)
        memory_result = self.tools.recall_customer_memory(
            str(ticket.get("customer_id", "anonymous")),
            state["category"],
            ticket_id=ticket_id,
        )
        knowledge = knowledge_result.get("results", [])
        memory = memory_result.get("memories", [])
        knowledge_confidence = float(knowledge_result.get("confidence", 0.0))

        route = state.get("route", "escalate")
        rationale = state.get("rationale", "")
        if knowledge_confidence < 0.25 or state["category"] == "general":
            route = "escalate"
            rationale += " Retrieval confidence or intent specificity is low, so a specialist should review."
        elif memory and state["category"] in {"billing", "refund", "cancellation"}:
            rationale += " Prior customer history was found and will personalize the response."

        return {
            "knowledge": knowledge,
            "memory_matches": memory,
            "knowledge_confidence": knowledge_confidence,
            "route": route,
            "rationale": rationale,
            "tools_used": state.get("tools_used", []) + ["search_knowledge", "recall_customer_memory"],
            "citations": [f"KB-{item['knowledge_id']}:{item['title']}" for item in knowledge],
            "actions_taken": state.get("actions_taken", []) + ["knowledge"],
        }

    def route_decision(self, state: TicketState) -> Route:
        return state.get("route", "escalate")

    def resolver_agent(self, state: TicketState) -> TicketState:
        category = state["category"]
        tools = list(state.get("tools_used", []))
        ticket = state["ticket"]
        ticket_id = state.get("ticket_id")
        citations = ", ".join(state.get("citations", [])) or "retrieved support policy"
        memory_note = ""
        if state.get("memory_matches"):
            memory_note = " I also considered the prior history for this customer."

        if category == "refund":
            validation = self.tools.validate_refund_request(ticket, ticket_id=ticket_id)
            tools.append("validate_refund_request")
            if validation.get("eligible"):
                response = (
                    "Your request appears eligible for standard refund review. "
                    f"I checked the refund policy and customer account before preparing this response. Source: {citations}.{memory_note}"
                )
            else:
                escalation = self.tools.create_escalation(ticket, category, "Refund tool validation failed", ticket_id=ticket_id)
                tools.append("create_escalation")
                return {
                    "route": "escalate",
                    "escalation_summary": str(escalation.get("summary")),
                    "response": "I am escalating this refund request because it needs specialist validation.",
                    "tools_used": tools,
                    "actions_taken": state.get("actions_taken", []) + ["resolver"],
                }
        elif category == "billing":
            invoice = self.tools.lookup_invoice(ticket, ticket_id=ticket_id)
            tools.append("lookup_invoice")
            response = (
                "I found the billing path for this request. "
                f"The next safe action is {invoice.get('invoice_action', 'billing specialist review')}. Source: {citations}.{memory_note}"
            )
        elif category == "cancellation":
            account = self.tools.get_account_status(str(ticket.get("customer_id", "")), ticket_id=ticket_id)
            tools.append("get_account_status")
            response = (
                "I can help begin cancellation. Please confirm the account id and acknowledge the data retention policy before final closure. "
                f"Account status checked: {(account.get('account') or {}).get('status', 'unknown')}. Source: {citations}.{memory_note}"
            )
        else:
            response = f"I found relevant support guidance and prepared a standard response. Source: {citations}.{memory_note}"
        return {"response": response, "tools_used": tools, "actions_taken": state.get("actions_taken", []) + ["resolver"]}

    def escalation_agent(self, state: TicketState) -> TicketState:
        ticket = state["ticket"]
        escalation = self.tools.create_escalation(ticket, state["category"], state["rationale"], ticket_id=state.get("ticket_id"))
        summary = str(escalation.get("summary"))
        citation_text = ", ".join(state.get("citations", [])) or "support policy search"
        response = (
            "I am escalating this to a specialist with the relevant context so it can be handled safely and quickly. "
            f"Source considered: {citation_text}."
        )
        return {
            "escalation_summary": summary,
            "response": response,
            "tools_used": state.get("tools_used", []) + ["create_escalation"],
            "actions_taken": state.get("actions_taken", []) + ["escalation"],
        }

    def memory_agent(self, state: TicketState) -> TicketState:
        ticket = state["ticket"]
        note = f"{state.get('route', 'unknown')}:{state['category']}:{state['response']}"
        self.tools.add_customer_memory(
            str(ticket.get("customer_id", "anonymous")),
            state["category"],
            note,
            ticket_id=state.get("ticket_id"),
        )
        return {
            "tools_used": state.get("tools_used", []) + ["add_customer_memory"],
            "actions_taken": state.get("actions_taken", []) + ["memory"],
        }


def demo_tickets() -> List[Dict[str, Any]]:
    return [
        {"ticket_id": "D-001", "customer_id": "C-100", "platform": "Zendesk", "urgency": "normal", "subject": "Refund request", "body": "I bought the annual plan last week and need a refund."},
        {"ticket_id": "D-002", "customer_id": "C-200", "platform": "Intercom", "urgency": "critical", "subject": "Production outage", "body": "Our workspace is down and API calls return 503 errors."},
        {"ticket_id": "D-003", "customer_id": "C-300", "platform": "Freshdesk", "urgency": "normal", "subject": "Need invoice", "body": "Please resend the April invoice to our billing email."},
        {"ticket_id": "D-004", "customer_id": "C-400", "platform": "Email", "urgency": "normal", "subject": "Cancel account", "body": "Please cancel my account after I export my data."},
        {"ticket_id": "D-005", "customer_id": "C-999", "platform": "Email", "urgency": "normal", "subject": "Need help", "body": "Something confusing happened and I am not sure what to do."},
    ]
