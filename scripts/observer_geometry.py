"""CLI entry point; notebook calculations live in occulted_flares.geometry.observer_geometry."""
from pathlib import Path
import sys
WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))

if __name__ == "__main__":
    import runpy
    runpy.run_module("occulted_flares.geometry.observer_geometry", run_name="__main__")
