"""Independent ray/sphere and limiting-case checks for occultation thresholds."""
import unittest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from sunpy.coordinates import frames
from occulted_flares.geometry.occultation_geometry import radial_occultation


class OccultationTest(unittest.TestCase):
    radius = 695700*u.km
    epoch = "2026-07-07T22:55:47.008"

    def observer(self, distance=0.73*u.AU):
        return SkyCoord(0*u.deg, 0*u.deg, distance,
                        frame=frames.HeliographicStonyhurst(obstime=self.epoch))

    def source(self, longitude):
        return SkyCoord(np.asarray(longitude)*u.deg, np.zeros_like(longitude)*u.deg, self.radius,
                        frame=frames.HeliographicStonyhurst(obstime=self.epoch))

    def test_threshold_is_tangent(self):
        obs = self.observer()
        for angle in (91, 93, 110):
            source = self.source(angle)
            height = float(radial_occultation(source, obs, self.radius)["h_occ_km"])
            direction = source.cartesian.xyz.to_value(u.km)/self.radius.value
            observer = obs.cartesian.xyz.to_value(u.km)
            for factor in (0.99, 1, 1.01):
                point = direction*(self.radius.value+height*factor)
                ray = observer-point
                t = -np.dot(point, ray)/np.dot(ray, ray)
                self.assertTrue(0 < t < 1)
                impact = np.linalg.norm(point+t*ray)
                if factor == 1:
                    self.assertAlmostEqual(impact, self.radius.value, places=6)
                elif factor < 1:
                    self.assertLess(impact, self.radius.value)
                else:
                    self.assertGreater(impact, self.radius.value)

    def test_visible_and_finite_limb(self):
        obs = self.observer()
        self.assertEqual(radial_occultation(self.source(0), obs, self.radius)["h_occ_km"], 0)
        # At alpha=90 the finite-distance observer still hides the photosphere.
        self.assertGreater(radial_occultation(self.source(90), obs, self.radius)["h_occ_km"], 0)

    def test_distant_limit(self):
        result = radial_occultation(self.source(95), self.observer(1e15*u.km), self.radius)
        expected = self.radius.value*(1/np.cos(np.deg2rad(5))-1)
        self.assertAlmostEqual(float(result["h_occ_km"]), expected, places=3)

    def test_vector_matches_scalar(self):
        angles = np.array([0., 91., 93., 110.])
        results = radial_occultation(self.source(angles), self.observer(), self.radius)
        for i, angle in enumerate(angles):
            scalar = radial_occultation(self.source(angle), self.observer(), self.radius)
            self.assertAlmostEqual(results["h_occ_km"][i], scalar["h_occ_km"], places=7)


if __name__ == "__main__":
    unittest.main()
