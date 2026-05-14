from __future__ import annotations

from agent import Agent, ECOHOME_SYSTEM_PROMPT
from evaluation import evaluate_response, evaluate_tool_usage, generate_evaluation_report
from models.energy import populate_sample_data, query_rows
from tools import get_electricity_prices, get_weather_forecast, search_energy_tips


def test_database_setup_contains_required_tables():
    populate_sample_data()
    usage = query_rows("SELECT COUNT(*) AS count FROM EnergyUsage")[0]["count"]
    solar = query_rows("SELECT COUNT(*) AS count FROM SolarGeneration")[0]["count"]
    assert usage > 0
    assert solar > 0


def test_external_tools_and_rag_tool_work():
    weather = get_weather_forecast.invoke("tomorrow")
    prices = get_electricity_prices.invoke("tomorrow")
    tips = search_energy_tips.invoke("EV charging solar")
    assert weather["forecast"]
    assert prices["hourly_prices"]
    assert tips["results"]


def test_agent_end_to_end_and_evaluation():
    agent = Agent(instructions=ECOHOME_SYSTEM_PROMPT)
    result = agent.invoke("How much can I save by running my dishwasher during off-peak hours?")
    assert "final_response" in result
    assert "calculate_savings" in result["tools_used"]
    response_eval = evaluate_response(
        "How much can I save by running my dishwasher during off-peak hours?",
        result["final_response"],
        "Estimated savings from moving dishwasher to off peak hours.",
    )
    tool_eval = evaluate_tool_usage(
        result["tools_used"],
        ["search_energy_tips", "get_electricity_prices", "calculate_savings"],
    )
    report = generate_evaluation_report([
        {"response_evaluation": response_eval, "tool_evaluation": tool_eval}
    ])
    assert response_eval["overall"] > 0.5
    assert tool_eval["tool_completeness"] >= 1.0
    assert report["num_tests"] == 1
