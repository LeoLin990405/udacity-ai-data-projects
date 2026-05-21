# Architecture and Decision Flow

## Named Pattern

The solution uses a **supervisor-routed specialist workflow**. A supervisor prepares a normalized state, a classifier decides the support category and initial risk, a retrieval agent grounds the work in database knowledge and customer memory, and a conditional route sends the ticket to either a resolver or an escalation specialist.

## Inputs and Outputs

Input is a text ticket represented as a dictionary with `customer_id`, `subject`, `body`, `urgency`, and optional metadata such as `platform` or `ticket_id`. Output is a structured state containing the route, category, confidence values, retrieved knowledge citations, tools used, customer response, escalation summary when needed, and the ordered list of agent actions.

## Agent Responsibilities

- The supervisor agent normalizes the ticket text and keeps the original metadata available for downstream tools.
- The classifier agent identifies the category, urgency, initial confidence, and high-risk cases such as outages or account access issues.
- The knowledge retrieval agent searches the SQLite `Knowledge` table and recalls persisted `CustomerMemory` records for the same customer and category.
- The resolver agent handles routine grounded cases with database-backed tools such as refund validation, invoice lookup, and account status checks.
- The escalation agent creates a structured specialist handoff when urgency, risk, or retrieval confidence indicates that automation should not decide alone.
- The memory agent persists the final outcome so later tickets can use customer history.

## Database and Tools

The setup creates the required `Account`, `User`, `Ticket`, `TicketMetadata`, `TicketMessage`, and `Knowledge` tables. It also adds `CustomerMemory` for long-term memory and `ToolLog` for searchable structured logs. The knowledge base is seeded with 15 articles across refund, billing, account access, technical outage, cancellation, and general triage categories.

Tools are callable methods on `SupportTools`. Each tool validates its inputs, queries or updates SQLite, returns a structured `success` or `error` payload, and writes a JSON record to `ToolLog`. This makes the workflow reproducible and auditable.

## Routing Logic

The classifier escalates high urgency, account access, and outage cases by default. The knowledge retrieval step can also change the route to escalation if the top retrieved article confidence is low. Routine refund, billing, and cancellation requests are resolved only after relevant articles are retrieved and the resolver has called the proper database-backed tool. Prior customer memory is included in the rationale and response personalization when available.

## Mermaid Diagram

See `docs/workflow_diagram.mmd` for the visual workflow diagram included with the submission.
