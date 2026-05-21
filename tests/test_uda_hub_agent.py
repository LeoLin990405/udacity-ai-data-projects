import unittest

from uda_hub_agent import UDAHubAgent


class UDAHubAgentTest(unittest.TestCase):
    def test_refund_ticket_is_resolved_and_remembered(self):
        agent = UDAHubAgent()
        ticket = {"customer_id": "C-1", "urgency": "normal", "subject": "Refund", "body": "Please refund my eligible order."}
        first = agent.invoke(ticket, thread_id="C-1")
        second = agent.invoke(ticket, thread_id="C-1")
        self.assertEqual(first["route"], "resolve")
        self.assertEqual(first["category"], "refund")
        self.assertIn("refund_policy_check", first["tools_used"])
        self.assertTrue(second["memory_matches"])

    def test_critical_outage_is_escalated(self):
        agent = UDAHubAgent()
        ticket = {"customer_id": "C-2", "urgency": "critical", "subject": "Outage", "body": "Production is down with 503 errors."}
        result = agent.invoke(ticket, thread_id="C-2")
        self.assertEqual(result["route"], "escalate")
        self.assertEqual(result["category"], "technical_outage")
        self.assertIn("create_escalation_summary", result["tools_used"])
        self.assertIn("Escalate", result["escalation_summary"])

    def test_billing_ticket_has_citation_and_action_trace(self):
        agent = UDAHubAgent()
        ticket = {"customer_id": "C-3", "urgency": "normal", "subject": "Invoice", "body": "Can you resend my invoice?"}
        result = agent.invoke(ticket, thread_id="C-3")
        self.assertEqual(result["route"], "resolve")
        self.assertTrue(result["citations"])
        self.assertEqual(result["actions_taken"][-1], "memory")


if __name__ == "__main__":
    unittest.main()
