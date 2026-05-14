from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import get_all_tools


class EnergyAdvisorState(TypedDict, total=False):
    question: str
    context: dict[str, Any]
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: dict[str, Any]
    tools_used: list[str]
    final_response: str
    decision_log: list[str]


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


def _invoke(tool, arg):
    return tool.invoke(arg)


def _extract_kwh(question: str, default: float = 10.0) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*kwh", question.lower())
    return float(match.group(1)) if match else default


def _plan_tools(state: EnergyAdvisorState) -> EnergyAdvisorState:
    question = state["question"].lower()
    tools = ["search_energy_tips"]
    if any(word in question for word in ["weather", "solar", "tomorrow", "week", "wednesday", "forecast"]):
        tools.append("get_weather_forecast")
        tools.append("get_solar_generation")
    if any(word in question for word in ["price", "cost", "save", "savings", "off-peak", "peak", "charge", "run"]):
        tools.append("get_electricity_prices")
    if any(word in question for word in ["history", "usage", "reduce", "device", "hvac", "thermostat", "ev", "car", "dishwasher", "pool", "laundry"]):
        device = "all"
        for candidate in ["ev_charger", "hvac", "dishwasher", "pool_pump", "laundry"]:
            if candidate.replace("_", " ") in question or candidate.split("_")[0] in question:
                device = candidate
        if "thermostat" in question:
            device = "hvac"
        tools.append(("get_energy_usage", device))
    if any(word in question for word in ["save", "savings", "off-peak", "dishwasher", "charge"]):
        tools.append(("calculate_savings", {"current_hour": 18, "optimized_hour": 12, "kwh": _extract_kwh(question, 12.0)}))
    return {"decision_log": [f"Planned tools for question: {tools}"], "tool_results": {"planned": tools}}


def _run_tools(state: EnergyAdvisorState) -> EnergyAdvisorState:
    tools = _tool_map(get_all_tools())
    results: dict[str, Any] = {}
    used: list[str] = []
    for item in state.get("tool_results", {}).get("planned", []):
        if isinstance(item, tuple):
            name, arg = item
        else:
            name, arg = item, state["question"]
        try:
            results[name] = _invoke(tools[name], arg)
            used.append(name)
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)}
    return {
        "tool_results": results,
        "tools_used": used,
        "decision_log": [f"Executed tools: {used}"],
    }


def _compose_response(state: EnergyAdvisorState) -> EnergyAdvisorState:
    question = state["question"]
    results = state.get("tool_results", {})
    lines = [
        "EcoHome Energy Advisor recommendation",
        f"Question: {question}",
        "",
        "Data retrieved:",
    ]
    for name in state.get("tools_used", []):
        lines.append(f"- {name}: used successfully")

    prices = results.get("get_electricity_prices", {}).get("hourly_prices", [])
    solar = results.get("get_solar_generation", {}).get("best_solar_hours", [])
    weather = results.get("get_weather_forecast", {}).get("forecast", [])
    savings = results.get("calculate_savings")
    tips = results.get("search_energy_tips", {}).get("results", [])

    recommendation_bits = []
    if solar:
        best_hours = ", ".join(row["hour"] for row in solar[:3])
        recommendation_bits.append(f"Prioritize solar-heavy hours around {best_hours}:00 for flexible loads.")
    if prices:
        cheapest = min(prices, key=lambda row: row["price_per_kwh"])
        peak = max(prices, key=lambda row: row["price_per_kwh"])
        recommendation_bits.append(
            f"Use hour {cheapest['hour']}:00 when possible at ${cheapest['price_per_kwh']:.2f}/kWh and avoid hour {peak['hour']}:00 at ${peak['price_per_kwh']:.2f}/kWh."
        )
    if weather:
        recommendation_bits.append(
            f"Weather outlook is {weather[0]['condition']} with best solar window {weather[0]['best_solar_window']}."
        )
    if savings:
        recommendation_bits.append(
            f"Estimated savings for the schedule shift: ${savings['estimated_savings']:.2f}, with about {savings['co2_kg_avoided']} kg CO2 avoided."
        )
    if not recommendation_bits:
        recommendation_bits.append("Review recent usage and shift flexible loads toward lower-price or higher-solar periods.")

    lines.extend(["", "Recommendation:"])
    lines.extend(f"- {bit}" for bit in recommendation_bits)
    if tips:
        lines.extend(["", "Relevant knowledge-base citations:"])
        for tip in tips[:3]:
            lines.append(f"- {tip['id']}: {tip['document'][:180].strip()}...")
    lines.extend(["", "Decision log:"])
    lines.extend(f"- {entry}" for entry in state.get("decision_log", []))

    response = "\n".join(lines)
    return {"final_response": response, "messages": [AIMessage(content=response)], "decision_log": ["Composed final recommendation."]}


def create_graph():
    graph = StateGraph(EnergyAdvisorState)
    graph.add_node("plan_tools", _plan_tools)
    graph.add_node("run_tools", _run_tools)
    graph.add_node("compose_response", _compose_response)
    graph.set_entry_point("plan_tools")
    graph.add_edge("plan_tools", "run_tools")
    graph.add_edge("run_tools", "compose_response")
    graph.add_edge("compose_response", END)
    return graph.compile()


@dataclass
class Agent:
    instructions: str
    model: str = "gpt-4o-mini"
    graph: Any = field(default_factory=create_graph)

    def __post_init__(self):
        load_dotenv()
        self.llm = self._build_llm()

    def _build_llm(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            kwargs = {"model": self.model, "temperature": 0, "api_key": api_key}
            if os.getenv("OPENAI_BASE_URL"):
                kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
            return ChatOpenAI(**kwargs)
        except Exception:
            return None

    def invoke(self, question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        initial_state = {
            "question": question,
            "context": context or {},
            "messages": [HumanMessage(content=question)],
            "decision_log": [self.instructions[:160]],
        }
        return self.graph.invoke(initial_state)


ECOHOME_SYSTEM_PROMPT = """
You are EcoHome Energy Advisor, an AI agent for smart-home energy optimization.
Your role is to analyze weather forecasts, electricity pricing, solar generation,
historical device usage, and energy-saving best practices to provide personalized,
data-driven recommendations.

Steps:
1. Understand whether the user asks about EV charging, HVAC, appliances, solar,
   costs, savings, environmental impact, or usage history.
2. Retrieve the relevant data using weather, pricing, solar, usage, savings, and
   energy-tip tools.
3. Recommend a schedule or strategy that minimizes cost, maximizes solar usage,
   and preserves user comfort.
4. Cite relevant knowledge-base tips and include concrete savings or CO2 estimates
   when available.

Example questions include EV charging optimization, thermostat settings during
price spikes, appliance scheduling, pool-pump timing, and reducing usage based on
history.
"""
