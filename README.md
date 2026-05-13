# Beaver's Choice Paper Company Multi-Agent Sales Team

This repository contains a completed Udacity Agentic AI project for the
Beaver's Choice Paper Company. The system uses a four-agent architecture built
with `smolagents` to process customer paper-supply requests, check inventory,
generate quotes, decide on reorders, and finalize sales transactions.

## Submission Contents

- `workflow_diagram.png`: Visual workflow diagram for the multi-agent system.
- `workflow_diagram.md`: Mermaid source and agent/tool responsibility table.
- `project_starter.py`: The single Python implementation file.
- `reflection_report.txt`: Detailed system explanation, evaluation summary, and improvement plan.
- `test_results.csv`: Full evaluation output over `quote_requests_sample.csv`.
- `quote_requests_sample.csv`: Provided sample requests used for evaluation.
- `quotes.csv`: Historical quote data used by the quoting agent.
- `requirements.txt`: Python dependencies.

## Agent Architecture

The implementation uses four agents, below the maximum of five:

- `OrchestratorAgent`: Parses customer requests, delegates to workers, and creates the final customer-facing response.
- `InventoryAgent`: Checks stock, evaluates reorder needs, and records replenishment transactions.
- `QuotingAgent`: Searches historical quotes and applies bulk-discount pricing.
- `SalesAgent`: Records fulfilled sales and produces financial snapshots.

The required starter helper functions are all wrapped as agent tools:

- `create_transaction`
- `get_all_inventory`
- `get_stock_level`
- `get_supplier_delivery_date`
- `get_cash_balance`
- `generate_financial_report`
- `search_quote_history`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file if you want to rerun the live agent:

```text
UDACITY_OPENAI_API_KEY=your_vocareum_key_here
UDACITY_OPENAI_API_BASE=https://openai.vocareum.com/v1
UDACITY_OPENAI_MODEL=gpt-4o-mini
```

The project does not commit API keys or local `.env` files.

## Run

```bash
python project_starter.py
```

The script initializes the SQLite database, processes all rows in
`quote_requests_sample.csv`, writes `test_results.csv`, and prints a final
financial report.

## Evaluation Snapshot

The submitted `test_results.csv` was generated from the full 20-row sample
request set.

- Total requests processed: 20.
- Cash-balance state changes: 18.
- Responses with successful fulfilled quote/order content: 18.
- Responses with unfulfilled or partial-fulfillment reasoning: 12.

This demonstrates that the system changes business state, fulfills multiple
requests, and also declines or partially fulfills requests when constraints
require it.

