"""Analytic surface-visibility checks, independent of online ephemerides."""
import unittest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from sunpy.coordinates import frames
from occulted_flares.geometry.observer_geometry import project_surface


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.epoch = "2026-07-07T22:55:47.008"
        self.radius = 695700*u.km
        self.observer = SkyCoord(0*u.deg, 0*u.deg, 1*u.AU,
                                 frame=frames.HeliographicStonyhurst(obstime=self.epoch))

    def source(self, lon):
        return SkyCoord(lon*u.deg, 0*u.deg, self.radius,
                        frame=frames.HeliographicStonyhurst(obstime=self.epoch))

    def test_center_visible(self):
        p = project_surface(self.source(0), self.observer)
        self.assertTrue(p["surface_visible"])
        self.assertAlmostEqual(p["tx_arcsec"], 0, places=6)
        self.assertAlmostEqual(p["ty_arcsec"], 0, places=6)

    def test_far_side_hidden(self):
        self.assertFalse(project_surface(self.source(180), self.observer)["surface_visible"])
        self.assertFalse(project_surface(self.source(90), self.observer)["surface_visible"])

    def test_finite_distance_limb(self):
        tangent = np.degrees(np.arccos((self.radius/self.observer.radius).decompose().value))
        result = project_surface(self.source(tangent), self.observer)
        self.assertAlmostEqual(result["surface_limb_margin_deg"], 0, places=7)
        self.assertTrue(project_surface(self.source(tangent-0.1), self.observer)["surface_visible"])
        self.assertFalse(project_surface(self.source(tangent+0.1), self.observer)["surface_visible"])


if __name__ == "__main__":
    unittest.main()
