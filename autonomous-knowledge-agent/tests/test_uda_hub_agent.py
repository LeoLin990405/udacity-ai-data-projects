import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from uda_hub_agent import UDAHubAgent


class UDAHubAgentTest(unittest.TestCase):
    def make_agent(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        return UDAHubAgent(db_path=Path(self.tmp.name) / "uda_hub.db")

    def test_refund_ticket_is_resolved_and_remembered(self):
        agent = self.make_agent()
        ticket = {"customer_id": "C-100", "urgency": "normal", "subject": "Refund", "body": "Please refund my eligible order."}
        first = agent.invoke(ticket, thread_id="C-100")
        second = agent.invoke(ticket, thread_id="C-100")
        self.assertEqual(first["route"], "resolve")
        self.assertEqual(first["category"], "refund")
        self.assertIn("validate_refund_request", first["tools_used"])
        self.assertTrue(second["memory_matches"])
        self.assertGreaterEqual(first["knowledge_confidence"], 0.25)

    def test_critical_outage_is_escalated(self):
        agent = self.make_agent()
        ticket = {"customer_id": "C-200", "urgency": "critical", "subject": "Outage", "body": "Production is down with 503 errors."}
        result = agent.invoke(ticket, thread_id="C-200")
        self.assertEqual(result["route"], "escalate")
        self.assertEqual(result["category"], "technical_outage")
        self.assertIn("create_escalation", result["tools_used"])
        self.assertIn("Escalate", result["escalation_summary"])

    def test_billing_ticket_has_citation_and_action_trace(self):
        agent = self.make_agent()
        ticket = {"customer_id": "C-300", "urgency": "normal", "subject": "Invoice", "body": "Can you resend my invoice?"}
        result = agent.invoke(ticket, thread_id="C-300")
        self.assertEqual(result["route"], "resolve")
        self.assertTrue(result["citations"])
        self.assertEqual(result["actions_taken"][-1], "memory")
        self.assertIn("lookup_invoice", result["tools_used"])

    def test_unknown_intent_escalates_with_low_retrieval_confidence(self):
        agent = self.make_agent()
        ticket = {"customer_id": "C-404", "urgency": "normal", "subject": "Odd request", "body": "The screen is purple and I need something different."}
        result = agent.invoke(ticket, thread_id="C-404")
        self.assertEqual(result["route"], "escalate")
        self.assertIn("low", result["rationale"].lower())


if __name__ == "__main__":
    unittest.main()
