from uda_hub_agent import UDAHubAgent, demo_tickets


if __name__ == "__main__":
    agent = UDAHubAgent()
    for ticket in demo_tickets():
        result = agent.invoke(ticket, thread_id=ticket["customer_id"])
        print("=" * 72)
        print("ticket:", ticket["subject"])
        print("route:", result["route"])
        print("category:", result["category"])
        print("confidence:", result["confidence"])
        print("tools:", ", ".join(result["tools_used"]))
        print("citations:", ", ".join(result.get("citations", [])))
        print("response:", result["response"])
        if result.get("escalation_summary"):
            print("escalation:", result["escalation_summary"])
