# EcoHome Energy Advisor

This is the submitted `ecohome_solution/` project for the EcoHome Energy Advisor. It implements a LangGraph-powered smart-home energy optimization agent with database tools, weather and electricity-price mock tools, ChromaDB RAG over energy-saving tips, savings calculations, and evaluation utilities.

## Python and Dependencies

Use Python 3.10 or newer. The project was tested locally with Python 3.14. Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core packages are:

- `langgraph`
- `langchain-core`
- `langchain-openai`
- `chromadb`
- `pydantic`
- `python-dotenv`
- `pytest`

## Setup

Run the notebooks in order:

1. `01_db_setup.ipynb` initializes SQLite and populates sample `EnergyUsage` and `SolarGeneration` data.
2. `02_rag_setup.ipynb` initializes ChromaDB and indexes the energy-saving documents under `data/documents/`.
3. `03_run_and_evaluate.ipynb` defines the system prompt, runs at least 10 test cases, records tool usage, and generates an evaluation report.

You can also run the demo script:

```bash
python run_demo.py
```

## Agent Contract

`agent.py` defines:

- `Agent(instructions, model="gpt-4o-mini")`
- `Agent.invoke(question, context=None)`

The agent uses LangGraph with this flow:

```text
plan_tools -> run_tools -> compose_response -> END
```

The agent response includes:

- data retrieval summary
- tool usage
- recommendation
- savings and CO2 estimates when relevant
- citations from the energy-saving knowledge base
- decision log

## Tools

`tools.py` defines:

- `get_weather_forecast`
- `get_electricity_prices`
- `get_energy_usage`
- `get_solar_generation`
- `search_energy_tips`
- `calculate_savings`

The weather and electricity pricing tools use deterministic mock data suitable for reproducible local grading. `search_energy_tips` uses ChromaDB with a deterministic local embedding function so it can run without external API calls. During initialization, `_load_or_create_collection()` explicitly loads documents, splits them into overlapping text chunks, stores chunk metadata, and reuses the persistent vector store on later runs.

## Evaluation

`evaluation.py` implements:

- `evaluate_response()` with accuracy, relevance, completeness, usefulness, and feedback.
- `evaluate_tool_usage()` with tool appropriateness, tool completeness, missing tools, extra tools, and feedback. It accepts the raw agent `messages` trace and extracts tool names internally from `AIMessage.tool_calls`.
- `generate_evaluation_report()` for aggregate metrics, strengths, weaknesses, and improvement recommendations.
- `display_evaluation_report()` for readable report output.

## Tests

```bash
pytest -q
```

The tests verify database setup, external/mock tools, RAG retrieval, agent workflow, savings calculation, and evaluation metrics.
