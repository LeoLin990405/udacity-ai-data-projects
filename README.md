# UDA-Hub Autonomous Knowledge Agent

UDA-Hub is a LangGraph-powered multi-agent decision suite for customer support automation. It classifies incoming tickets, retrieves internal knowledge, chooses a route, resolves routine cases, escalates risky cases, and stores reusable long-term memory.

## Architecture

The workflow is intentionally small but complete:

1. `supervisor_agent` normalizes the incoming ticket and metadata.
2. `classifier_agent` assigns category, urgency, confidence, and routing decision.
3. `knowledge_agent` retrieves relevant FAQ snippets and prior memory.
4. `resolver_agent` drafts a customer-safe resolution for routine tickets.
5. `escalation_agent` prepares an internal escalation summary for high-risk or low-confidence tickets.
6. `memory_agent` stores resolution notes for future related tickets.

The implementation uses `StateGraph`, conditional edges, and a checkpointer when LangGraph is installed. A deterministic fallback executor is included so the project can still be tested in restricted local environments.

## Run

```bash
python run_demo.py
python -m pytest -q
```

## Example Capabilities

- Refund requests under policy are resolved with the refund tool.
- Urgent outages and identity-sensitive account access issues are escalated.
- FAQ retrieval cites matched knowledge snippets.
- Long-term memory stores prior outcomes by customer id and category.
- The final response includes route, confidence, decision rationale, tools used, citations, and customer-facing answer.
