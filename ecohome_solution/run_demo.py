from __future__ import annotations

from agent import Agent, ECOHOME_SYSTEM_PROMPT
from evaluation import (
    display_evaluation_report,
    evaluate_response,
    evaluate_tool_usage,
    generate_evaluation_report,
)
from models.energy import populate_sample_data
from tools import search_energy_tips


def main() -> None:
    populate_sample_data()
    search_energy_tips.invoke("solar appliance scheduling")
    agent = Agent(instructions=ECOHOME_SYSTEM_PROMPT)
    test_cases = [
        {
            "question": "When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
            "expected": "Use midday solar hours and avoid evening peak prices.",
            "tools": ["search_energy_tips", "get_weather_forecast", "get_solar_generation", "get_electricity_prices", "get_energy_usage", "calculate_savings"],
        },
        {
            "question": "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?",
            "expected": "Use HVAC pre-cooling and adjust thermostat during peak-price periods.",
            "tools": ["search_energy_tips", "get_weather_forecast", "get_solar_generation", "get_electricity_prices", "get_energy_usage"],
        },
        {
            "question": "Suggest three ways I can reduce energy use based on my usage history.",
            "expected": "Review usage history and shift flexible loads to off-peak or solar periods.",
            "tools": ["search_energy_tips", "get_energy_usage"],
        },
    ]
    results = []
    for case in test_cases:
        output = agent.invoke(case["question"])
        results.append(
            {
                "question": case["question"],
                "response": output["final_response"],
                "tools_used": output["tools_used"],
                "response_evaluation": evaluate_response(case["question"], output["final_response"], case["expected"]),
                "tool_evaluation": evaluate_tool_usage(output["tools_used"], case["tools"]),
            }
        )
    report = generate_evaluation_report(results)
    print(display_evaluation_report(report))
    for result in results:
        print("\nQUESTION:", result["question"])
        print("TOOLS:", result["tools_used"])
        print(result["response"])


if __name__ == "__main__":
    main()
