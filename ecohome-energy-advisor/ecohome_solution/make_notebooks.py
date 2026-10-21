from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = cells
    nbf.write(nb, ROOT / name)


write_notebook(
    "01_db_setup.ipynb",
    [
        md(
            "# 01 Database Setup\n\n"
            "This notebook initializes the EcoHome SQLite database and populates it with reproducible sample data for energy usage and solar generation. "
            "The project rubric requires the database to contain `EnergyUsage` and `SolarGeneration` tables before the agent is evaluated."
        ),
        code(
            "from models.energy import initialize_database, populate_sample_data, query_rows\n"
            "initialize_database()\n"
            "populate_sample_data()\n"
            "print('EnergyUsage rows:', query_rows('SELECT COUNT(*) AS count FROM EnergyUsage')[0]['count'])\n"
            "print('SolarGeneration rows:', query_rows('SELECT COUNT(*) AS count FROM SolarGeneration')[0]['count'])"
        ),
        md("## Energy Usage Summary\n\nThis table shows historical device-level usage and cost totals used by the recommendation agent."),
        code(
            "query_rows('SELECT device_type, ROUND(SUM(consumption_kwh), 2) AS kwh, ROUND(SUM(cost_usd), 2) AS cost FROM EnergyUsage GROUP BY device_type ORDER BY cost DESC')"
        ),
        md("## Solar Generation Summary\n\nThis aggregation verifies that weather-conditioned solar generation data exists for analysis."),
        code(
            "query_rows('SELECT weather_condition, ROUND(AVG(generation_kwh), 2) AS avg_generation FROM SolarGeneration GROUP BY weather_condition')"
        ),
        md("## Sample Rows\n\nThe following rows demonstrate the exact schema content available to the tools."),
        code(
            "print('EnergyUsage sample:')\n"
            "for row in query_rows('SELECT * FROM EnergyUsage LIMIT 3'):\n"
            "    print(row)\n"
            "print('\\nSolarGeneration sample:')\n"
            "for row in query_rows('SELECT * FROM SolarGeneration LIMIT 3'):\n"
            "    print(row)"
        ),
    ],
)

write_notebook(
    "02_rag_setup.ipynb",
    [
        md(
            "# 02 RAG Setup\n\n"
            "This notebook verifies that the knowledge base contains the starter documents plus at least five additional energy-saving documents. "
            "It then creates or loads the ChromaDB vector store used by the `search_energy_tips` tool."
        ),
        code(
            "from pathlib import Path\n"
            "docs = sorted(Path('data/documents').glob('*.txt'))\n"
            "print('Knowledge base documents:', len(docs))\n"
            "for path in docs:\n"
            "    print('-', path.name)"
        ),
        md("## Knowledge Base Topics\n\nThe added documents cover HVAC, automation, renewable integration, seasonal management, and storage optimization."),
        code(
            "from pathlib import Path\n"
            "for path in sorted(Path('data/documents').glob('*.txt')):\n"
            "    text = path.read_text(encoding='utf-8').strip().replace('\\n', ' ')\n"
            "    print(f'{path.name}: {text[:220]}...')"
        ),
        md("## RAG Search Smoke Test\n\nThe output below demonstrates that ChromaDB returns relevant documents with source metadata."),
        code(
            "from tools import search_energy_tips\n"
            "result = search_energy_tips.invoke('solar EV charging HVAC storage optimization')\n"
            "result"
        ),
        code(
            "for hit in result['results']:\n"
            "    print(hit['id'], hit['metadata'])\n"
            "    print(hit['document'][:300].replace('\\n', ' '), '\\n')"
        ),
    ],
)

test_cases_code = """
from agent import Agent, ECOHOME_SYSTEM_PROMPT
from evaluation import evaluate_response, evaluate_tool_usage, generate_evaluation_report, display_evaluation_report
from models.energy import populate_sample_data
from tools import search_energy_tips

populate_sample_data()
search_energy_tips.invoke('initialize vector database for solar energy tips')

ecohome_agent = Agent(
    instructions=ECOHOME_SYSTEM_PROMPT,
)

test_cases = [
    {
        'id': 'EV-001',
        'question': 'When should I charge my electric car tomorrow to minimize cost and maximize solar power?',
        'expected_response': 'Charge during midday solar hours and avoid evening peak prices.',
        'expected_tools': ['search_energy_tips', 'get_weather_forecast', 'get_solar_generation', 'get_electricity_prices', 'get_energy_usage', 'calculate_savings'],
    },
    {
        'id': 'HVAC-002',
        'question': 'What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?',
        'expected_response': 'Pre-cool before peak prices and raise the thermostat by 2 to 4 degrees during spikes.',
        'expected_tools': ['search_energy_tips', 'get_weather_forecast', 'get_solar_generation', 'get_electricity_prices', 'get_energy_usage'],
    },
    {
        'id': 'HISTORY-003',
        'question': 'Suggest three ways I can reduce energy use based on my usage history.',
        'expected_response': 'Use history to identify high-cost devices and shift flexible loads.',
        'expected_tools': ['search_energy_tips', 'get_energy_usage'],
    },
    {
        'id': 'APPLIANCE-004',
        'question': 'How much can I save by running my dishwasher during off-peak hours?',
        'expected_response': 'Move dishwasher from peak to off peak and estimate savings.',
        'expected_tools': ['search_energy_tips', 'get_electricity_prices', 'get_energy_usage', 'calculate_savings'],
    },
    {
        'id': 'POOL-005',
        'question': 'What is the best time to run my pool pump this week based on the weather forecast?',
        'expected_response': 'Run the pool pump during solar production and lower price windows.',
        'expected_tools': ['search_energy_tips', 'get_weather_forecast', 'get_solar_generation', 'get_electricity_prices', 'get_energy_usage'],
    },
    {
        'id': 'BATTERY-006',
        'question': 'Should I use my home battery during evening peak pricing?',
        'expected_response': 'Discharge battery during peak evening pricing while preserving backup capacity.',
        'expected_tools': ['search_energy_tips', 'get_electricity_prices'],
    },
    {
        'id': 'SOLAR-007',
        'question': 'How can I maximize self-consumption of solar power with laundry and appliances?',
        'expected_response': 'Shift appliances to midday solar windows and automate flexible loads.',
        'expected_tools': ['search_energy_tips', 'get_weather_forecast', 'get_solar_generation', 'get_energy_usage'],
    },
    {
        'id': 'EV-008',
        'question': 'Compare EV charging at 6 PM versus noon for a 12 kWh session.',
        'expected_response': 'Noon is cheaper and aligns with solar; estimate savings for 12 kWh.',
        'expected_tools': ['search_energy_tips', 'get_electricity_prices', 'get_solar_generation', 'get_energy_usage', 'calculate_savings'],
    },
    {
        'id': 'HVAC-009',
        'question': 'What seasonal strategy should I use for summer HVAC costs?',
        'expected_response': 'Pre-cool, use blinds, and avoid afternoon peak HVAC operation.',
        'expected_tools': ['search_energy_tips', 'get_weather_forecast', 'get_energy_usage'],
    },
    {
        'id': 'HISTORY-010',
        'question': 'Which devices have the highest historical energy cost?',
        'expected_response': 'Query usage history and identify the highest-cost devices.',
        'expected_tools': ['search_energy_tips', 'get_energy_usage'],
    },
]

test_results = []
for case in test_cases:
    result = ecohome_agent.invoke(case['question'])
    response_eval = evaluate_response(case['question'], result['final_response'], case['expected_response'])
    tool_eval = evaluate_tool_usage(result['messages'], case['expected_tools'])
    test_results.append({
        'id': case['id'],
        'question': case['question'],
        'final_response': result['final_response'],
        'tools_used': result['tools_used'],
        'messages_list': result['messages'],
        'response_evaluation': response_eval,
        'tool_evaluation': tool_eval,
    })

len(test_results), test_results[0]
"""

write_notebook(
    "03_run_and_evaluate.ipynb",
    [
        md(
            "# 03 Run and Evaluate\n\n"
            "This notebook crafts `ECOHOME_SYSTEM_PROMPT`, instantiates the LangGraph agent, runs 10 test cases, stores transparency data in `test_results`, and generates an evaluation report. "
            "The test cases cover EV charging, thermostat settings, appliances, solar maximization, battery use, historical usage, and cost savings."
        ),
        md("## System Prompt\n\nThe prompt below defines the Energy Advisor's role, steps, capabilities, recommendation style, and example questions."),
        code("from agent import ECOHOME_SYSTEM_PROMPT\nprint(ECOHOME_SYSTEM_PROMPT)"),
        md("## End-to-End Evaluation\n\nThis cell runs the agent over 10 scenarios. Each `test_results` item includes the question, final response, tools used, response metrics, and tool-usage metrics."),
        code(test_cases_code),
        md("## Aggregate Evaluation Report\n\nThe report calculates overall scores, strengths, weaknesses, and recommendations for improvement."),
        code(
            "report = generate_evaluation_report(test_results)\n"
            "print(display_evaluation_report(report))"
        ),
        md("## Example Agent Outputs\n\nThese examples show data retrieval, tool usage, citations, recommendations, and decision logs in the final answer."),
        code(
            "for item in test_results[:3]:\n"
            "    print('\\nQUESTION:', item['question'])\n"
            "    print('TOOLS:', item['tools_used'])\n"
            "    print(item['final_response'][:900])"
        ),
        md("## Tool Coverage Table\n\nThis compact table makes it easy for reviewers to verify that each required tool is exercised."),
        code(
            "coverage = []\n"
            "for item in test_results:\n"
            "    coverage.append({\n"
            "        'id': item['id'],\n"
            "        'question': item['question'],\n"
            "        'tools_used': ', '.join(item['tools_used']),\n"
            "        'response_score': item['response_evaluation']['overall'],\n"
            "        'tool_completeness': item['tool_evaluation']['tool_completeness'],\n"
            "    })\n"
            "for row in coverage:\n"
            "    print(row)\n"
            "coverage"
        ),
    ],
)

print("Notebooks written.")
