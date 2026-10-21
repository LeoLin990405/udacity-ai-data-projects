# UdaPlay: AI Research Agent for Video Games

UdaPlay is a small agentic RAG system for answering natural-language questions
about video games. It first searches an internal ChromaDB knowledge base built
from local JSON files. If internal confidence is low, it falls back to Tavily
web search, persists the discovered facts to long-term memory, and returns a
structured cited answer.

## Project Files

- `Udaplay_01_solution_project.ipynb` builds the local game vector store and demonstrates semantic search.
- `Udaplay_02_solution_project.ipynb` runs the stateful research agent on example queries.
- `src/udaplay/vector_store.py` processes JSON data, embeds records, and manages ChromaDB.
- `src/udaplay/agent.py` implements `retrieve_game`, `evaluate_retrieval`, `game_web_search`, and the state machine.
- `data/games/*.json` contains the local knowledge base.
- `memory/learned_facts.jsonl` stores web-search results learned by the agent.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with:

```text
OPENAI_API_KEY="your-openai-key"
TAVILY_API_KEY="your-tavily-key"
```

The project includes a deterministic local embedding fallback so the notebooks
and smoke tests run without keys. With keys present, the embedding manager uses
OpenAI `text-embedding-3-small` and Tavily web search.

## Run

```bash
python scripts/build_vector_store.py
python scripts/run_examples.py
```

The example run demonstrates:

1. An internal answer: `Who developed FIFA 21?`
2. An internal answer: `When was God of War Ragnarok released?`
3. A fallback-style query: `What is Rockstar Games working on right now?`

## Rubric Coverage

- Local game JSON files are loaded, formatted, embedded, and persisted in ChromaDB.
- The vector store can be queried with semantic search.
- The agent has three integrated tools: internal retrieval, retrieval evaluation, and web search.
- The agent maintains conversation state and records tool usage for each query.
- Long-term memory persists learned web-search facts in JSONL format.
- Outputs include citations, confidence, reasoning, and structured JSON.

