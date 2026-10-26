"""Run notebook startup, data checks, and first fit without changing the notebook."""
import faulthandler
import sys
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
print("Loading notebook verification tools", flush=True)
import nbformat
from nbclient import NotebookClient

work = Path(__file__).resolve().parents[1]
path = work / "SOL2026-07-07/Spectroscopy/notebooks/03_First_reference_fit.ipynb"
notebook = nbformat.read(path, as_version=4)
notebook.cells = notebook.cells[:6]
print("Starting registered stix-research kernel", flush=True)
client = NotebookClient(
    notebook, kernel_name="stix-research", timeout=180,
    resources={"metadata": {"path": str(path.parent)}},
    on_cell_start=lambda cell, cell_index: print(
        f"Cell {cell_index}: {cell.cell_type}", flush=True),
)
client.execute()
faulthandler.cancel_dump_traceback_later()
print("PASS: kernel startup, notebook initialization, STIX data checks, first fit", flush=True)
print("Original notebook and scientific output files were not rewritten.", flush=True)
