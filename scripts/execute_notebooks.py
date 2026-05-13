#!/usr/bin/env python3
"""Execute notebooks so submitted files include visible outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name="udaplay-venv", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}")


def main() -> None:
    execute(ROOT / "Udaplay_01_solution_project.ipynb")
    execute(ROOT / "Udaplay_02_solution_project.ipynb")


if __name__ == "__main__":
    main()
