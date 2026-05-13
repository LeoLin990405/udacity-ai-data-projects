#!/usr/bin/env python3
"""Create the two required Udacity solution notebooks."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def write_notebook(path: Path, cells) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "UdaPlay venv", "language": "python", "name": "udaplay-venv"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, path)


def main() -> None:
    notebook_1 = [
        nbf.v4.new_markdown_cell("# UdaPlay 01 Solution Project: RAG Pipeline\n\nThis notebook loads local video game JSON files, creates embeddings, persists them in ChromaDB, and demonstrates semantic search."),
        nbf.v4.new_code_cell("from dotenv import load_dotenv\nload_dotenv('.env')\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('src').resolve()))"),
        nbf.v4.new_code_cell("from udaplay import VectorStoreManager\nfrom udaplay.vector_store import format_results\n\nmanager = VectorStoreManager(data_dir='data/games', chroma_dir='chroma_db')\ndocuments = manager.load_game_documents()\nlen(documents), documents[0]"),
        nbf.v4.new_markdown_cell("## Build Persistent ChromaDB Collection\n\nThe embedding provider uses OpenAI when an API key is configured and otherwise falls back to deterministic local embeddings for reproducible offline execution."),
        nbf.v4.new_code_cell("count = manager.build_index(reset=True)\nprint(f'Indexed {count} game records')\nprint('Collection size:', manager.collection.count())"),
        nbf.v4.new_markdown_cell("## Semantic Search Demo"),
        nbf.v4.new_code_cell("results = manager.query('Who developed FIFA 21?', top_k=3)\nprint(format_results(results))"),
        nbf.v4.new_code_cell("results = manager.query('What platform was Pokemon Red launched on?', top_k=3)\nprint(format_results(results))"),
    ]
    write_notebook(ROOT / "Udaplay_01_solution_project.ipynb", notebook_1)

    notebook_2 = [
        nbf.v4.new_markdown_cell("# UdaPlay 02 Solution Project: Agent Implementation\n\nThis notebook demonstrates a stateful research agent with internal retrieval, retrieval evaluation, Tavily web fallback, persistent memory, and structured cited answers."),
        nbf.v4.new_code_cell("from dotenv import load_dotenv\nload_dotenv('.env')\nimport json, sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('src').resolve()))"),
        nbf.v4.new_code_cell("from udaplay import UdaPlayAgent, VectorStoreManager\n\nmanager = VectorStoreManager(data_dir='data/games', chroma_dir='chroma_db')\nif manager.collection.count() == 0:\n    manager.build_index(reset=True)\nagent = UdaPlayAgent(vector_store=manager, memory_path='memory/learned_facts.jsonl')"),
        nbf.v4.new_markdown_cell("## Tool Demonstration\n\nThe agent exposes the required tools: `retrieve_game`, `evaluate_retrieval`, and `game_web_search`."),
        nbf.v4.new_code_cell("retrieved = agent.retrieve_game('When was God of War Ragnarok released?')\nprint(json.dumps(agent.evaluate_retrieval('When was God of War Ragnarok released?', retrieved), indent=2))"),
        nbf.v4.new_markdown_cell("## Example Queries and Reports"),
        nbf.v4.new_code_cell("queries = [\n    'Who developed FIFA 21?',\n    'When was God of War Ragnarok released?',\n    'What platform was Pokémon Red launched on?',\n    'What is Rockstar Games working on right now?'\n]\nfor query in queries:\n    report = agent.answer(query)\n    print('\\n' + '=' * 80)\n    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))"),
        nbf.v4.new_markdown_cell("## Conversation State\n\nThe agent remembers prior turns and stores learned web facts in long-term memory."),
        nbf.v4.new_code_cell("agent.state.conversation[-4:]\n"),
    ]
    write_notebook(ROOT / "Udaplay_02_solution_project.ipynb", notebook_2)
    print("Created Udaplay_01_solution_project.ipynb and Udaplay_02_solution_project.ipynb")


if __name__ == "__main__":
    main()
