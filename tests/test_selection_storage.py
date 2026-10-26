"""Verify stable point storage does not erase selections on cancel/failure."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from occulted_flares.geometry import euvi_coordinates
class StorageTest(unittest.TestCase):
    def check(self, mode):
        script = Path(__file__).resolve().parent.parent / "scripts/select_points.py"
        spec = importlib.util.spec_from_file_location("event_picker", script)
        picker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(picker)
        with tempfile.TemporaryDirectory() as folder:
            geometry = Path(folder)/"Geometry"
            geometry.mkdir()
            (geometry/"frame.fts").write_bytes(b"test input")
            (geometry/"select_points.json").write_text(json.dumps(dict(input_file="frame.fts", output_root="selections")))
            current = geometry/"selections/current"
            current.mkdir(parents=True)
            names = ["selected_pixels.csv", "coordinates.csv", "metadata.json", "selected_points.png"]
            for name in names:
                (current/name).write_bytes(b"previous selection")

            def fake_run(args):
                staged = Path(args[args.index("--output")+1])
                staged.mkdir()
                if mode == "failure":
                    return 1
                (staged/"selected_pixels.csv").write_text("file,label,x_pixel,y_pixel\n" + ("frame.fts,point_1,1,2\n" if mode == "success" else ""))
                (staged/"coordinates.csv").write_text("new coordinates")
                (staged/"metadata.json").write_text("{}")
                (staged/"image.png").write_bytes(b"new image")
                return 0

            with patch.object(picker, "__file__", str(geometry/"select_points.py")), patch.object(euvi_coordinates, "main", side_effect=fake_run):
                result = picker.main()
            self.assertEqual(result, 1 if mode == "failure" else 0)
            if mode == "success":
                self.assertEqual((current/"coordinates.csv").read_text(), "new coordinates")
                for name in names:
                    self.assertEqual((geometry/"selections/previous"/name).read_bytes(), b"previous selection")
            else:
                for name in names:
                    self.assertEqual((current/name).read_bytes(), b"previous selection")
                self.assertFalse((geometry/"selections/previous").exists())

    def test_success_keeps_one_previous(self):
        self.check("success")

    def test_empty_keeps_current(self):
        self.check("empty")

    def test_failure_keeps_current(self):
        self.check("failure")


if __name__ == "__main__":
    unittest.main()
