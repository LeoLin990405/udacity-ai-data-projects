from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


QA_SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Answer only from the retrieved context. Cite document IDs in the sources field.
If the context is insufficient, say what is missing and lower confidence.
"""

SUMMARIZATION_SYSTEM_PROMPT = """You are a document summarization assistant.
Create concise but useful summaries from the retrieved context. Extract key points,
include source document IDs, and avoid adding facts that are not in the context.
"""

CALCULATION_SYSTEM_PROMPT = """You are a document calculation assistant.
For every calculation request:
1. Determine which document must be retrieved and inspect the retrieved context.
2. Determine the mathematical expression needed to answer the user's request.
3. Use the calculator tool for ALL calculations, even simple arithmetic.
4. Return the expression, result, explanation, confidence, and source document IDs.
Do not do mental math in the answer; rely on the calculator tool result.
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """Classify the user's request into exactly one intent:
- qa: asks for a specific fact, value, detail, comparison, or explanation from documents.
- summarization: asks for a summary, overview, key points, highlights, or executive brief.
- calculation: asks to compute, calculate, add, subtract, multiply, divide, compare totals, rates, percentages, differences, or ratios.
- unknown: cannot be classified.

Use the conversation history only as context. Return a valid structured object with intent_type,
confidence, and reasoning.
"""


def get_intent_classification_prompt() -> ChatPromptTemplate:
    """Return the prompt used by classify_intent."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            (
                "human",
                "Conversation history:\n{conversation_history}\n\nCurrent user input:\n{user_input}",
            ),
        ]
    )


def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """Return a role-specific chat prompt template."""

    prompts = {
        "qa": QA_SYSTEM_PROMPT,
        "summarization": SUMMARIZATION_SYSTEM_PROMPT,
        "calculation": CALCULATION_SYSTEM_PROMPT,
    }
    system_prompt = prompts.get(intent_type, QA_SYSTEM_PROMPT)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Conversation history:\n{conversation_history}\n\n"
                "Retrieved context:\n{context}\n\n"
                "User request:\n{user_input}",
            ),
        ]
    )
