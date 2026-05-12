"""Integration test: execute every notebook end-to-end.

Marked ``slow``. Each notebook is executed in-place with a 600 s timeout.
The notebook supplies its own calculator (EMT or EAM) in its first code
cell, so this test has no notebook-specific setup.
"""

from __future__ import annotations

import pathlib

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.slow
def test_notebook_runs(nb_path: pathlib.Path):
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
