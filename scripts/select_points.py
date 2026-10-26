"""Run this event's interactive point selection with VS Code's play button.

Edit select_points.json to choose a different FITS. This wrapper supplies event
settings; the reusable coordinate calculations remain in src/occulted_flares/geometry/euvi_coordinates.py.
"""
import csv
import json
from pathlib import Path
import sys
import shutil
import tempfile


def selection_arguments(output):
    geometry = Path(__file__).resolve().parent
    settings = json.loads((geometry / "select_points.json").read_text(encoding="utf-8"))
    fits_path = (geometry / settings["input_file"]).resolve()
    if not fits_path.is_file():
        raise FileNotFoundError(f"Selected FITS not found: {fits_path}")
    return [str(fits_path), "--pick", "--output", str(output)]


def main():
    geometry = Path(__file__).resolve().parent
    sys.path.insert(0, str(geometry.parent / "src"))
    from occulted_flares.geometry.euvi_coordinates import main as coordinates_main
    print("Opening the configured FITS frame for exploratory point measurements.")
    print("Click compact features/extent points; press Enter to save. Right click removes the last point.")
    settings = json.loads((geometry / "select_points.json").read_text())
    root = geometry / settings["output_root"]
    current, previous = root / "current", root / "previous"
    # Temporary staging is removed after the run. Cancelled/failed selection
    # must not erase the accepted points. Retain only one previous snapshot.
    with tempfile.TemporaryDirectory(prefix="euvi_points_") as temporary:
        staged = Path(temporary) / "selection"
        status = coordinates_main(selection_arguments(staged))
        if status:
            print("Selection failed; current selection retained.")
            return status
        with (staged / "selected_pixels.csv").open(newline="") as stream:
            count = len(list(csv.DictReader(stream)))
        if not count:
            print("No points selected; current selection retained.")
            return 0
        names = ["selected_pixels.csv", "coordinates.csv", "metadata.json", "selected_points.png"]
        if current.exists():
            previous.mkdir(parents=True, exist_ok=True)
            for name in names:
                source = current / name
                if source.exists():
                    shutil.copy2(source, previous / name)
        current.mkdir(parents=True, exist_ok=True)
        for name in names[:-1]:
            shutil.copy2(staged / name, current / name)
        shutil.copy2(next(staged.glob("*.png")), current / "selected_points.png")
        print(f"Saved {count} points to {current}; one prior selection retained in {previous}")
        print("Check selection_groups.json labels if the number or meaning of points changed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
