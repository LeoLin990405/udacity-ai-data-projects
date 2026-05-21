from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DB_PATH = Path("data/uda_hub.db")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS Account (
    account_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS User (
    user_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES Account(account_id)
);

CREATE TABLE IF NOT EXISTS Ticket (
    ticket_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT,
    priority TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES Account(account_id)
);

CREATE TABLE IF NOT EXISTS TicketMetadata (
    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY(ticket_id) REFERENCES Ticket(ticket_id)
);

CREATE TABLE IF NOT EXISTS TicketMessage (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(ticket_id) REFERENCES Ticket(ticket_id)
);

CREATE TABLE IF NOT EXISTS Knowledge (
    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS CustomerMemory (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    category TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ToolLog (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    tool_name TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    success INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


KNOWLEDGE_ARTICLES = [
    ("Refund eligibility window", "refund", "Refunds are available within 30 days for verified eligible purchases unless policy abuse is detected.", "refund,policy,eligibility"),
    ("Refund amount threshold", "refund", "Standard automated refunds may be approved for verified orders under five hundred dollars.", "refund,amount,automation"),
    ("Refund escalation triggers", "refund", "Escalate refunds involving chargebacks, fraud concerns, enterprise contracts, or missing ownership verification.", "refund,escalation,fraud"),
    ("Billing invoice resend", "billing", "Invoices can be resent to the billing email on file after account and requester validation.", "billing,invoice,email"),
    ("Billing tax correction", "billing", "Tax ID, VAT, address, and legal entity corrections should be routed to billing operations.", "billing,tax,operations"),
    ("Duplicate charge handling", "billing", "Duplicate charge reports require payment lookup, transaction comparison, and a billing specialist if amounts differ.", "billing,duplicate,charge"),
    ("Password reset self-service", "account_access", "For password issues, send the password reset link and advise checking spam and corporate email filters.", "password,login,account"),
    ("MFA recovery", "account_access", "If multi-factor authentication is unavailable, escalate to identity verification and do not bypass security controls.", "mfa,security,identity"),
    ("Account ownership mismatch", "account_access", "If the requester is not an admin or owner, provide safe guidance and request authorization from an account owner.", "account,owner,security"),
    ("Production outage intake", "technical_outage", "For production outages, collect workspace id, timestamps, affected users, error messages, and recent deployment details.", "outage,technical,incident"),
    ("Critical incident escalation", "technical_outage", "Critical outages and 5xx errors affecting production customers must be escalated immediately to the incident queue.", "outage,critical,escalation"),
    ("API rate limit guidance", "technical_outage", "Rate limit errors should include endpoint, request volume, response codes, and retry behavior before escalation.", "api,rate-limit,technical"),
    ("Cancellation confirmation", "cancellation", "Cancellation requires account id, cancellation reason, and confirmation that the customer understands retention policy.", "cancellation,retention"),
    ("Retention and exports", "cancellation", "Before final cancellation, customers should be offered export guidance and retention timelines for their plan.", "cancellation,export,retention"),
    ("General support triage", "general", "When intent is unclear, ask one concise clarifying question and avoid taking irreversible actions.", "general,triage,clarify"),
]


SEED_ACCOUNTS = [
    ("A-100", "Northwind Labs", "Pro", "active"),
    ("A-200", "Contoso Media", "Enterprise", "active"),
    ("A-300", "Fabrikam Finance", "Team", "past_due"),
    ("A-400", "Tailspin Games", "Starter", "active"),
]


SEED_USERS = [
    ("C-100", "A-100", "owner@northwind.example", "owner"),
    ("C-200", "A-200", "sre@contoso.example", "admin"),
    ("C-300", "A-300", "billing@fabrikam.example", "billing_admin"),
    ("C-400", "A-400", "player@tailspin.example", "member"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path | str = DB_PATH, reset: bool = False) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()

    with get_connection(path) as conn:
        conn.executescript(SCHEMA_SQL)
        seed_database(conn)
    return path


def seed_database(conn: sqlite3.Connection) -> None:
    now = utc_now()
    conn.executemany(
        "INSERT OR IGNORE INTO Account(account_id, name, plan, status, created_at) VALUES (?, ?, ?, ?, ?)",
        [(account_id, name, plan, status, now) for account_id, name, plan, status in SEED_ACCOUNTS],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO User(user_id, account_id, email, role) VALUES (?, ?, ?, ?)",
        SEED_USERS,
    )
    conn.execute("DELETE FROM Knowledge")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'Knowledge'")
    conn.executemany(
        "INSERT OR IGNORE INTO Knowledge(title, category, body, tags) VALUES (?, ?, ?, ?)",
        KNOWLEDGE_ARTICLES,
    )


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


@dataclass
class SupportTools:
    db_path: Path = DB_PATH

    def _log(self, tool_name: str, payload: Dict[str, Any], result: Dict[str, Any], ticket_id: Optional[str] = None) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ToolLog(ticket_id, tool_name, input_json, output_json, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    tool_name,
                    json.dumps(payload, sort_keys=True),
                    json.dumps(result, sort_keys=True),
                    1 if result.get("success") else 0,
                    utc_now(),
                ),
            )

    def get_account_status(self, customer_id: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"customer_id": customer_id}
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT a.account_id, a.name, a.plan, a.status, u.role
                FROM User u JOIN Account a ON a.account_id = u.account_id
                WHERE u.user_id = ?
                """,
                (customer_id,),
            ).fetchone()
        result = {"success": bool(row), "account": dict(row) if row else None}
        if not row:
            result["error"] = "customer_not_found"
        self._log("get_account_status", payload, result, ticket_id)
        return result

    def search_knowledge(self, query: str, category: str, ticket_id: Optional[str] = None, limit: int = 3) -> Dict[str, Any]:
        payload = {"query": query, "category": category, "limit": limit}
        terms = {term.strip(".,!?;:()[]").lower() for term in query.split() if len(term) > 2}
        with get_connection(self.db_path) as conn:
            rows = rows_to_dicts(conn.execute("SELECT * FROM Knowledge").fetchall())

        scored = []
        for article in rows:
            text = f"{article['title']} {article['category']} {article['body']} {article['tags']}".lower()
            term_score = sum(1 for term in terms if term in text)
            category_boost = 3 if article["category"] == category else 0
            score = term_score + category_boost
            if score:
                item = dict(article)
                item["score"] = score
                item["confidence"] = min(0.99, round(score / 8, 2))
                scored.append(item)

        scored.sort(key=lambda item: (item["score"], item["knowledge_id"]), reverse=True)
        results = scored[:limit]
        max_score = results[0]["score"] if results else 0
        result = {
            "success": bool(results),
            "results": results,
            "max_score": max_score,
            "confidence": min(0.99, round(max_score / 8, 2)) if max_score else 0.0,
        }
        if not results:
            result["error"] = "no_relevant_articles"
        self._log("search_knowledge", payload, result, ticket_id)
        return result

    def recall_customer_memory(self, customer_id: str, category: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"customer_id": customer_id, "category": category}
        with get_connection(self.db_path) as conn:
            rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT note, created_at FROM CustomerMemory
                    WHERE customer_id = ? AND category = ?
                    ORDER BY created_at DESC LIMIT 5
                    """,
                    (customer_id, category),
                ).fetchall()
            )
        result = {"success": True, "memories": rows}
        self._log("recall_customer_memory", payload, result, ticket_id)
        return result

    def validate_refund_request(self, ticket: Dict[str, Any], ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"ticket": ticket}
        account = self.get_account_status(str(ticket.get("customer_id", "")), ticket_id=ticket_id)
        text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
        eligible = account.get("success") and "chargeback" not in text and "fraud" not in text
        result = {
            "success": bool(eligible),
            "eligible": bool(eligible),
            "reason": "verified_customer_and_no_risk_terms" if eligible else "requires_specialist_review",
        }
        self._log("validate_refund_request", payload, result, ticket_id)
        return result

    def lookup_invoice(self, ticket: Dict[str, Any], ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"ticket": ticket}
        account = self.get_account_status(str(ticket.get("customer_id", "")), ticket_id=ticket_id)
        result = {
            "success": bool(account.get("success")),
            "invoice_action": "resend_to_billing_email" if account.get("success") else None,
            "account": account.get("account"),
        }
        if not account.get("success"):
            result["error"] = "account_lookup_failed"
        self._log("lookup_invoice", payload, result, ticket_id)
        return result

    def create_escalation(self, ticket: Dict[str, Any], category: str, rationale: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"ticket": ticket, "category": category, "rationale": rationale}
        result = {
            "success": True,
            "queue": "incident" if category == "technical_outage" else "specialist",
            "summary": (
                f"Escalate {category} ticket for customer {ticket.get('customer_id', 'unknown')}. "
                f"Urgency={ticket.get('urgency', 'normal')}. Reason: {rationale}. "
                f"Customer text: {ticket.get('subject', '')} {ticket.get('body', '')}".strip()
            ),
        }
        self._log("create_escalation", payload, result, ticket_id)
        return result

    def persist_ticket_and_message(self, ticket: Dict[str, Any], category: str) -> str:
        ticket_id = str(ticket.get("ticket_id") or f"T-{datetime.now(timezone.utc).timestamp():.6f}")
        customer_id = str(ticket.get("customer_id", "anonymous"))
        account = self.get_account_status(customer_id, ticket_id=None)
        account_id = ((account.get("account") or {}).get("account_id")) or "A-100"
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO Ticket(ticket_id, account_id, customer_id, subject, status, category, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    account_id,
                    customer_id,
                    str(ticket.get("subject", "")),
                    str(ticket.get("status", "open")),
                    category,
                    str(ticket.get("urgency", "normal")),
                    utc_now(),
                ),
            )
            conn.execute(
                "INSERT INTO TicketMessage(ticket_id, sender, body, created_at) VALUES (?, ?, ?, ?)",
                (ticket_id, "customer", str(ticket.get("body", "")), utc_now()),
            )
            for key in ("platform", "urgency", "channel"):
                if ticket.get(key) is not None:
                    conn.execute(
                        "INSERT INTO TicketMetadata(ticket_id, key, value) VALUES (?, ?, ?)",
                        (ticket_id, key, str(ticket[key])),
                    )
        return ticket_id

    def add_customer_memory(self, customer_id: str, category: str, note: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"customer_id": customer_id, "category": category, "note": note}
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO CustomerMemory(customer_id, category, note, created_at) VALUES (?, ?, ?, ?)",
                (customer_id, category, note, utc_now()),
            )
        result = {"success": True}
        self._log("add_customer_memory", payload, result, ticket_id)
        return result
