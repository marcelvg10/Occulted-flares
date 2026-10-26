"""Independent ray/segment checks of elevated-source boundary circles."""
import unittest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from sunpy.coordinates import frames
from occulted_flares.geometry.visibility_boundaries import boundary_coordinates


class BoundaryTest(unittest.TestCase):
    radius = 695700*u.km

    def observer(self):
        return SkyCoord(63*u.deg, 7*u.deg, 0.73*u.AU,
                        frame=frames.HeliographicStonyhurst(obstime="2026-07-07T22:55:47.008"))

    def test_full_circles_are_tangent_and_bracket_visibility(self):
        obs = self.observer()
        observer = obs.cartesian.xyz.to_value(u.km)
        for height in (1000, 50000, 100000):
            roots, sources, _ = boundary_coordinates(obs, self.radius, height*u.km, samples=128)
            unit = roots.cartesian.xyz.to_value(u.km)/self.radius.value
            for factor in (0.99, 1, 1.01):
                points = unit*(self.radius.value+height*factor)
                ray = observer[:, None]-points
                t = -np.sum(points*ray, axis=0)/np.sum(ray*ray, axis=0)
                self.assertTrue(np.all((t > 0) & (t < 1)))
                impact = np.linalg.norm(points+ray*t, axis=0)
                if factor == 1:
                    np.testing.assert_allclose(impact, self.radius.value, rtol=0, atol=1e-7)
                elif factor < 1:
                    self.assertTrue(np.all(impact < self.radius.value))
                else:
                    self.assertTrue(np.all(impact > self.radius.value))

    def test_zero_height_is_finite_distance_surface_limb(self):
        obs = self.observer()
        root, source, angle = boundary_coordinates(obs, self.radius, 0*u.km, samples=128)
        n = root.cartesian.xyz.to_value(u.km)/self.radius.value
        dot = np.sum(n*obs.cartesian.xyz.to_value(u.km)[:, None], axis=0)
        np.testing.assert_allclose(dot, self.radius.value, rtol=0, atol=1e-7)
        self.assertAlmostEqual(angle, np.rad2deg(np.arccos((self.radius/obs.radius).decompose().value)))

    def test_circle_closes_and_radius_is_fixed(self):
        root, source, _ = boundary_coordinates(self.observer(), self.radius, 50*u.Mm, samples=128)
        np.testing.assert_allclose(root.cartesian.xyz[:, 0].value, root.cartesian.xyz[:, -1].value, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(root.cartesian.xyz.value, axis=0), self.radius.value)
        np.testing.assert_allclose(np.linalg.norm(source.cartesian.xyz.value, axis=0), self.radius.value+50000)

    def test_invalid_height_rejected(self):
        with self.assertRaises(ValueError):
            boundary_coordinates(self.observer(), self.radius, -1*u.km)


if __name__ == "__main__":
    unittest.main()
