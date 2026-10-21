from __future__ import annotations

from src.assistant import AssistantSession, DocumentAssistant


def main() -> None:
    assistant = DocumentAssistant(session=AssistantSession(session_id="demo-session"))
    prompts = [
        "What was Q2 revenue?",
        "Summarize the healthcare quality report.",
        "Calculate the difference between 1410000 and 1250000.",
    ]
    for prompt in prompts:
        result = assistant.process_message(prompt)
        print("\nUSER:", prompt)
        print("INTENT:", result["intent"].intent_type)
        print("ACTIONS:", result["actions_taken"])
        print("RESPONSE:", result["current_response"])


if __name__ == "__main__":
    main()
