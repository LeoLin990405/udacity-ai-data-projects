from __future__ import annotations

from pprint import pprint

from src.assistant import DocumentAssistant


def main() -> None:
    assistant = DocumentAssistant()
    print("Document Assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        result = assistant.process_message(user_input)
        pprint(result.get("current_response", result))


if __name__ == "__main__":
    main()
