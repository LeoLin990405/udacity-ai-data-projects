# Document Assistant Project

This repository implements the Udacity Document Assistant project with LangChain and LangGraph. The assistant classifies user intent, retrieves document context, routes requests to Q&A, summarization, or calculation agents, logs tool use, and persists session history.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For Vocareum keys, set:

```env
OPENAI_API_KEY=voc-...
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

If no key is present, the project uses `FakeDocumentLLM` so the workflow and tests can still run locally.

## Run

```bash
python main.py
```

Example questions:

- `What was Q2 revenue?`
- `Summarize the healthcare quality report.`
- `Calculate the difference between Q2 revenue and Q1 revenue.`

## Implementation Notes

- `src/schemas.py` defines Pydantic structured outputs for `AnswerResponse`, `UserIntent`, `SummaryResponse`, `CalculationResponse`, and memory updates.
- `src/agent.py` defines the LangGraph `AgentState` with `add_messages` for chat history and `operator.add` for `actions_taken`.
- `create_workflow()` builds the graph: `classify_intent -> qa/summarization/calculation -> update_memory -> END`, then compiles it with `InMemorySaver`.
- `src/tools.py` implements a logged `document_reader` tool and a safe `calculator` tool using `@tool`. The calculator validates AST nodes before evaluating expressions and always returns strings.
- `src/assistant.py` sets the required configurable values: `thread_id`, `llm`, and `tools`.

## Memory and Logs

The assistant writes one JSONL session file per conversation under `sessions/`. Tool calls are logged under `logs/` with timestamp, session ID, tool name, input, and output.

## Testing

```bash
pytest -q
```

The tests demonstrate:

- Schema validation and type enforcement.
- Intent routing for Q&A, summarization, and calculation.
- Safe calculator behavior and graceful error handling.
- Session and tool log generation.
