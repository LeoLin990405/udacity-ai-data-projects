# Beaver's Choice — Multi-Agent Workflow Diagram

## High-level architecture (4 agents, ≤5 cap)

```mermaid
flowchart TD
    CUST[Customer Request<br/>free-text + date]
    ORCH[OrchestratorAgent<br/>parses request<br/>delegates tasks<br/>composes response]
    INV[InventoryAgent<br/>stock checks<br/>reorder decisions]
    QUOTE[QuotingAgent<br/>price + bulk discount<br/>delivery quote]
    SALES[SalesAgent<br/>transaction recording<br/>finance reporting]

    DB[(munder_difflin.db<br/>SQLite)]
    HIST[(quotes.csv<br/>historical pricing)]

    CUST --> ORCH
    ORCH -->|"delegate_check_inventory(items, date)"| INV
    ORCH -->|"delegate_quote(items, qty, date)"| QUOTE
    ORCH -->|"delegate_finalize_sale(items, prices, date)"| SALES
    ORCH --> RESP[Customer-facing response<br/>quote + delivery + status]

    INV -->|"get_all_inventory(date)"| DB
    INV -->|"get_stock_level(item, date)"| DB
    INV -->|"create_transaction(stock_orders, date)"| DB

    QUOTE -->|"search_quote_history([keywords])"| HIST
    QUOTE -->|"get_supplier_delivery_date(date, qty)"| DB

    SALES -->|"create_transaction(sales, date)"| DB
    SALES -->|"get_cash_balance(date)"| DB
    SALES -->|"generate_financial_report(date)"| DB
```

## Agent responsibilities

| Agent | Inputs | Outputs | Tools (helpers used) |
|-------|--------|---------|---------------------|
| **OrchestratorAgent** | raw customer request + date | sanitized customer response | delegate_* tools wrapping the 3 worker agents |
| **InventoryAgent** | item list + qty + date | per-item availability ✅ / partial / reorder needed | `get_all_inventory`, `get_stock_level`, `get_supplier_delivery_date`, `create_transaction` (for stock-purchase events) |
| **QuotingAgent** | items + qty + date | unit price + bulk discount + final price + ETA | `search_quote_history`, `get_supplier_delivery_date` |
| **SalesAgent** | items + final prices + qty + date | transaction confirmation + post-sale cash balance | `create_transaction` (for sales), `get_cash_balance`, `generate_financial_report` |

All 7 mandated helper functions are exercised across the 4 agents.

## Decision logic

1. **OrchestratorAgent** parses the free-text request via prompt-engineered LLM (Chain-of-Thought + few-shot).
2. **InventoryAgent**: if `get_stock_level(item, date) >= qty` → fulfillable; if `< qty` → check `get_supplier_delivery_date` for restock window; if `== 0` and supplier ETA > customer deadline → mark unfulfillable.
3. **QuotingAgent**: base unit price from `paper_supplies`; bulk discount tiers (5% @ 500 units, 10% @ 1000, 15% @ 5000); `search_quote_history` to align with historical pricing.
4. **SalesAgent**: only creates a "sales" transaction for fulfilled lines; updates cash via `create_transaction(... transaction_type='sales' ...)`.
5. Customer response is composed by orchestrator — ordered breakdown + total + ETA + (if applicable) reasons for unfulfilled items.
