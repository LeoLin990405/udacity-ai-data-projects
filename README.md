# AgentsVille Trip Planner

> Udacity Agentic AI Nanodegree (nd900, Core #6: Intelligent Systems) — Course Project

## Overview

A multi-agent AI travel planning system combining:
- **ItineraryAgent** — generates travel plans via role-based prompting + Chain-of-Thought
- **ItineraryRevisionAgent** — optimizes plans via ReACT (THINK-ACT-OBSERVE) with 4 tools

## Techniques

- Role-Based Prompting, CoT reasoning, ReACT agent loop
- Pydantic structured output validation (`VacationInfo`, `Activity`, `DayPlan`, `TravelPlan`)
- 4 tools: `calculator_tool`, `get_activities_by_date_tool`, `run_evals_tool`, `final_answer_tool`
- LLM-driven evaluation (weather-activity compatibility + 5-dim multi-eval)

## Setup (any OpenAI-compatible endpoint)

```bash
pip install -r requirements.txt

# Option A — Vocareum proxy (if your Udacity program has Vocareum credits)
export OPENAI_API_KEY="voc-..."
export OPENAI_BASE_URL="https://openai.vocareum.com/v1"
export AGENTSVILLE_MODEL="gpt-4o-mini"

# Option B — Direct OpenAI
export OPENAI_API_KEY="sk-..."
export AGENTSVILLE_MODEL="gpt-4o-mini"

# Option C — Zhipu GLM-4.6 (this submission used this)
export OPENAI_API_KEY="$GLM_API_KEY"
export OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export AGENTSVILLE_MODEL="glm-4.6"
export AGENTSVILLE_DISABLE_THINKING="1"

jupyter nbconvert --to notebook --execute agentsville_trip_planner.ipynb \
    --output agentsville_trip_planner.executed.ipynb \
    --ExecutePreprocessor.timeout=900
```

## Model choice note

The reference notebook targeted `gpt-4o-mini`. This submission was executed with **Zhipu GLM-4.6** via its OpenAI-compatible endpoint because the current environment has a GLM Coding Plan subscription instead of a Vocareum/OpenAI key. Methodology is identical — GLM-4.6 exposes `chat.completions.create` with the same tool/message API; the only quirk is that thinking mode must be disabled (`extra_body={"thinking": {"type": "disabled"}}`) so the final `content` field is populated. The notebook supports all three providers via env vars (see Setup).

## Results (ReACT loop, 7 iterations, all rubric checks PASS)

### Plan Metrics
| Field | Value |
|---|---|
| Traveler | Alice Johnson |
| Destination | San Francisco |
| Dates | 2025-06-15 → 2025-06-18 (4 days) |
| Total cost | **$302.00** |
| Budget | $2000.00 |
| Activities | 13 across 4 days |

### Multi-dimensional evaluation (all PASS)
| Dimension | Score |
|---|---|
| `city_match` | PASS |
| `date_match` | PASS |
| `budget_adherence` | PASS |
| `activity_authenticity` | PASS |
| `weather_compatibility` | PASS |

### ReACT iteration sequence
1. `run_evals_tool` — baseline evaluation (all PASS)
2–6. `calculator_tool` × 5 — verify each day's cost + total ($75 + $125 + $52 + $50 = $302 ✓)
7. `final_answer_tool` — submit verified plan

Rubric requirement: agent must call `run_evals_tool` **before** `final_answer_tool` — satisfied (iteration 1 vs 7).

## Files

| File | Description |
|---|---|
| `agentsville_trip_planner.ipynb` | Complete executed notebook with all outputs |
| `requirements.txt` | `openai`, `pydantic`, `matplotlib`, `IPython` (no CUDA deps) |
