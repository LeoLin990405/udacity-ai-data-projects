#!/usr/bin/env python3
"""Build the UdaPlay ChromaDB vector store."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from udaplay import VectorStoreManager
from udaplay.vector_store import format_results


def main() -> None:
    manager = VectorStoreManager(data_dir=ROOT / "data/games", chroma_dir=ROOT / "chroma_db")
    count = manager.build_index(reset=True)
    print(f"Indexed {count} game records into {manager.collection_name}.")
    results = manager.query("Who developed FIFA 21?", top_k=2)
    print(format_results(results))


if __name__ == "__main__":
    main()
