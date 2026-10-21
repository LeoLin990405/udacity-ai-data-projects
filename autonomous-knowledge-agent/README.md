# UDA-Hub Autonomous Knowledge Agent

UDA-Hub is a LangGraph-powered multi-agent decision suite for customer support automation. It classifies incoming tickets, retrieves internal knowledge from SQLite, chooses a route, resolves routine cases with callable support tools, escalates risky cases, logs tool usage, and stores reusable long-term customer memory.

## Architecture

The workflow follows a **supervisor-routed specialist workflow** pattern:

1. `supervisor_agent` normalizes the incoming ticket and metadata.
2. `classifier_agent` assigns category, urgency, confidence, and routing decision.
3. `knowledge_agent` searches the `Knowledge` table, scores article relevance, and recalls persistent customer history.
4. `resolver_agent` drafts a customer-safe resolution for routine tickets using database-backed tools.
5. `escalation_agent` prepares an internal escalation summary for high-risk, sensitive, or low-confidence tickets.
6. `memory_agent` stores resolution notes in `CustomerMemory` for future related tickets.

The implementation uses `StateGraph`, conditional edges, and a checkpointer when LangGraph is installed. A deterministic fallback executor is included so the project can still be tested in restricted local environments.

The visual workflow is in `docs/workflow_diagram.mmd`, and the detailed design writeup is in `docs/architecture.md`.

## Database Setup

The required project tables are created by `database.py` and can be initialized from either the notebook or the setup script:

```bash
python setup_database.py
```

The schema includes:

- `Account`
- `User`
- `Ticket`
- `TicketMetadata`
- `TicketMessage`
- `Knowledge`
- `CustomerMemory`
- `ToolLog`

The `Knowledge` table is seeded with 15 articles, exceeding the 14-article minimum. The tool layer records structured JSON input/output for each tool call in `ToolLog`.

## Run

```bash
python setup_database.py
python run_demo.py
python -m pytest -q
```

## Example Capabilities

- Refund requests under policy are resolved after `validate_refund_request` checks account status and risk terms.
- Billing requests use `lookup_invoice` to choose the safe invoice action.
- Urgent outages and identity-sensitive account access issues are escalated using `create_escalation`.
- Retrieval cites matched `Knowledge` rows such as `KB-4:Billing invoice resend`.
- Long-term memory is persisted by customer id and category, then retrieved on later turns to influence rationale and personalization.
- The final response includes route, category, classifier confidence, retrieval confidence, decision rationale, tools used, citations, and customer-facing answer.

## Demo Coverage

`run_demo.py` includes five ticket scenarios:

- Routine refund request.
- Critical production outage.
- Billing invoice resend request.
- Cancellation/export request.
- Ambiguous low-confidence request that escalates.
