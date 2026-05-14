from __future__ import annotations

from statistics import mean
from typing import Any


def evaluate_response(question: str, final_response: str, expected_response: str) -> dict[str, Any]:
    """Score response quality for accuracy, relevance, completeness, and usefulness."""

    response_lower = final_response.lower()
    expected_terms = [term for term in expected_response.lower().replace(",", " ").split() if len(term) > 4]
    matched_terms = [term for term in expected_terms if term in response_lower]

    relevance = 1.0 if any(word in response_lower for word in question.lower().split() if len(word) > 4) else 0.7
    accuracy = min(1.0, len(matched_terms) / max(len(expected_terms), 1) + 0.25)
    completeness_signals = ["recommendation", "data retrieved", "citations", "savings", "decision log"]
    completeness = sum(signal in response_lower for signal in completeness_signals) / len(completeness_signals)
    usefulness = 1.0 if any(word in response_lower for word in ["avoid", "use", "prioritize", "estimated", "schedule"]) else 0.6

    scores = {
        "accuracy": round(accuracy, 3),
        "relevance": round(relevance, 3),
        "completeness": round(completeness, 3),
        "usefulness": round(usefulness, 3),
    }
    scores["overall"] = round(mean(scores.values()), 3)
    scores["feedback"] = (
        f"Matched {len(matched_terms)} expected terms. "
        f"Response includes {sum(signal in response_lower for signal in completeness_signals)} of {len(completeness_signals)} required sections/signals."
    )
    return scores


def evaluate_tool_usage(messages_list: list[str], expected_tools: list[str]) -> dict[str, Any]:
    """Evaluate whether the agent used appropriate and complete tools."""

    used = set(messages_list)
    expected = set(expected_tools)
    missing = sorted(expected - used)
    extra = sorted(used - expected)
    appropriateness = 1.0 if not extra else max(0.5, 1 - 0.1 * len(extra))
    completeness = len(expected & used) / max(len(expected), 1)
    return {
        "tool_appropriateness": round(appropriateness, 3),
        "tool_completeness": round(completeness, 3),
        "missing_tools": missing,
        "extra_tools": extra,
        "feedback": f"Used {sorted(used)}; expected {sorted(expected)}.",
    }


def generate_evaluation_report(test_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate aggregate evaluation report from test results."""

    response_scores = [item["response_evaluation"]["overall"] for item in test_results]
    tool_scores = [item["tool_evaluation"]["tool_completeness"] for item in test_results]
    strengths = []
    weaknesses = []
    if mean(tool_scores) >= 0.8:
        strengths.append("The agent consistently selected the expected data tools.")
    else:
        weaknesses.append("Some test cases missed expected tools.")
    if mean(response_scores) >= 0.75:
        strengths.append("Responses were generally relevant and actionable.")
    else:
        weaknesses.append("Responses need more precise expected-answer coverage.")
    recommendations = [
        "Add live weather and utility-price API integrations for production use.",
        "Add user preference learning for comfort and EV deadline constraints.",
        "Add charts for hourly cost and solar-alignment recommendations.",
    ]
    return {
        "overall_response_score": round(mean(response_scores), 3),
        "overall_tool_completeness": round(mean(tool_scores), 3),
        "num_tests": len(test_results),
        "strengths": strengths,
        "weaknesses": weaknesses or ["No major weaknesses observed in this test run."],
        "recommendations": recommendations,
    }


def display_evaluation_report(report: dict[str, Any]) -> str:
    lines = [
        "# EcoHome Evaluation Report",
        f"Tests run: {report['num_tests']}",
        f"Overall response score: {report['overall_response_score']}",
        f"Overall tool completeness: {report['overall_tool_completeness']}",
        "",
        "## Strengths",
        *[f"- {item}" for item in report["strengths"]],
        "",
        "## Weaknesses",
        *[f"- {item}" for item in report["weaknesses"]],
        "",
        "## Recommendations",
        *[f"- {item}" for item in report["recommendations"]],
    ]
    return "\n".join(lines)
