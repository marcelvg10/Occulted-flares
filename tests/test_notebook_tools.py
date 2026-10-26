"""Check selection controls and backup behavior without touching event inputs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import sunpy.map
from occulted_flares.geometry.notebook_tools import NotebookPointPicker, commit_selection


class NotebookToolsTest(unittest.TestCase):
    def setUp(self):
        observer = SkyCoord(50*u.deg, 5*u.deg, 1*u.AU,
                            frame=frames.HeliographicStonyhurst, obstime="2026-07-07T22:55:00")
        center = SkyCoord(0*u.arcsec, 0*u.arcsec, observer=observer,
                          obstime=observer.obstime, frame=frames.Helioprojective)
        header = sunpy.map.make_fitswcs_header((64, 64), center, scale=[5, 5]*u.arcsec/u.pix)
        yy, xx = np.indices((64, 64))
        self.m = sunpy.map.Map(np.exp(-((xx-31.5)**2+(yy-31.5)**2)/100.), header)

    def tearDown(self):
        plt.close("all")

    def test_click_add_remove_and_ignore_outside(self):
        picker = NotebookPointPicker(self.m)
        picker.on_click(SimpleNamespace(inaxes=None, xdata=31, ydata=32, button=1))
        self.assertFalse(picker.points)
        picker.on_click(SimpleNamespace(inaxes=picker.axes, xdata=31, ydata=32, button=1))
        self.assertEqual(picker.points[0]["x_pixel"], 31.)
        self.assertEqual(picker.points[0]["y_pixel"], 32.)
        picker.on_click(SimpleNamespace(inaxes=picker.axes, xdata=31, ydata=32, button=3))
        self.assertFalse(picker.points)
        picker.disconnect()

    def test_explicit_commit_preserves_previous_and_updates_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"fits").mkdir()
            path = root/"fits/test.fits"
            self.m.save(path)
            config = root/"selection_groups.json"
            old_config = dict(fits_directory="fits", selection_run="selections/current", groups={"old": ["old_label"]})
            config.write_text(json.dumps(old_config))
            current = root/"selections/current"
            current.mkdir(parents=True)
            (current/"coordinates.csv").write_bytes(b"previous coordinates")
            commit_selection(config, self.m, path, [dict(label="new", x_pixel=31, y_pixel=32)])
            self.assertEqual((root/"selections/previous/coordinates.csv").read_bytes(), b"previous coordinates")
            self.assertEqual(json.loads((root/"selections/previous/selection_groups.json").read_text()), old_config)
            self.assertEqual(json.loads(config.read_text())["groups"], {"A_upper_candidate": ["new"]})
            self.assertTrue((current/"selected_points.png").is_file())

    def test_empty_commit_keeps_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/"selection_groups.json"
            config.write_bytes(b"unchanged")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                commit_selection(config, self.m, Path(directory)/"missing.fits", [])
            self.assertEqual(config.read_bytes(), b"unchanged")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
