"""Offline synthetic checks; does not test EUVI calibration or real pointing."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from occulted_flares.geometry.euvi_coordinates import configure_cache, load_map, measure
configure_cache()

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np
from sunpy.coordinates import frames
import sunpy.map



class CoordinatesTest(unittest.TestCase):
    def setUp(self):
        observer = SkyCoord(50*u.deg, 5*u.deg, 1*u.AU,
                            frame=frames.HeliographicStonyhurst, obstime="2026-07-07T22:55:00")
        center = SkyCoord(0*u.arcsec, 0*u.arcsec, observer=observer,
                          obstime=observer.obstime, frame=frames.Helioprojective)
        header = sunpy.map.make_fitswcs_header((512, 512), center, scale=[5, 5]*u.arcsec/u.pix)
        y, x = np.indices((512, 512))
        data = np.exp(-((x-255.5)**2+(y-255.5)**2)/10000.)
        self.m = sunpy.map.Map(data, header)

    def test_surface_and_off_limb(self):
        row = measure(self.m, "center", 255.5, 255.5)
        self.assertAlmostEqual(row["hgs_lon_deg"], 50, places=6)
        self.assertAlmostEqual(row["hgs_lat_deg"], 5, places=6)
        self.assertEqual(measure(self.m, "off_limb", 0, 0)["status"], "off_limb_surface_undefined")
        self.assertEqual(measure(self.m, "outside", -1, 0)["status"], "outside_image")

    def test_missing_observer_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            header = self.m.fits_header
            for key in ("HGLN_OBS", "HGLT_OBS", "DSUN_OBS"):
                header.pop(key, None)
            path = Path(folder)/"missing_observer.fits"
            fits.writeto(path, self.m.data, header)
            with self.assertRaisesRegex(ValueError, "Observer metadata incomplete"):
                load_map(path, True)

    def test_cli_products(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"SYNTHETIC_NOT_EVENT.fits"
            self.m.save(path)
            points = root/"points.csv"
            points.write_text("file,label,x_pixel,y_pixel\nSYNTHETIC_NOT_EVENT.fits,center,255.5,255.5\nSYNTHETIC_NOT_EVENT.fits,off_limb,0,0\n")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/"scripts/euvi_coordinates.py"),
                                     str(path), "--output", str(root/"run"), "--points", str(points),
                                     "--allow-synthetic"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with (root/"run"/"coordinates.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(list((root/"run").glob("*.png")))
            metadata = json.loads((root/"run"/"metadata.json").read_text())
            self.assertTrue(metadata["synthetic_test"])


if __name__ == "__main__":
    unittest.main()
