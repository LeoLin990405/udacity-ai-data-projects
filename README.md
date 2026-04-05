# AgentsVille Trip Planner

> Udacity Agentic AI Nanodegree (Core #6: Intelligent Systems) — Course Project

## Overview

A multi-agent AI travel planning system combining:
- **ItineraryAgent**: Generates travel plans via role-based prompting + Chain-of-Thought
- **ItineraryRevisionAgent**: Optimizes plans via ReACT (THINK-ACT-OBSERVE) with tool calling

## Techniques
- Role-Based Prompting, CoT reasoning, ReACT agent loop
- Pydantic structured output validation
- LLM-driven evaluation (weather-activity compatibility)

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
jupyter notebook agentsville_trip_planner.ipynb
```
