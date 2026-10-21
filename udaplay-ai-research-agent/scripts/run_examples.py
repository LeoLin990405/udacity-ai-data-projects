#!/usr/bin/env python3
"""Run UdaPlay example queries."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from udaplay import UdaPlayAgent, VectorStoreManager


def main() -> None:
    manager = VectorStoreManager(data_dir=ROOT / "data/games", chroma_dir=ROOT / "chroma_db")
    manager.build_index(reset=True)

    agent = UdaPlayAgent(vector_store=manager, memory_path=ROOT / "memory/learned_facts.jsonl")
    questions = [
        "Who developed FIFA 21?",
        "When was God of War Ragnarok released?",
        "What platform was Pokémon Red launched on?",
        "What is Rockstar Games working on right now?",
    ]
    for question in questions:
        answer = agent.answer(question)
        print("=" * 80)
        print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
