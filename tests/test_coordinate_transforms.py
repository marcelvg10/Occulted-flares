"""Numerical and geometrical checks for full-3D observer transformations."""
import unittest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from sunpy.coordinates import frames
from occulted_flares.geometry.coordinate_transforms import source_coordinate, transform_with_check


class TransformTest(unittest.TestCase):
    epoch = "2026-07-07T22:55:47.008"

    def observer(self, lon, lat=0):
        return SkyCoord(lon*u.deg, lat*u.deg, 1*u.AU,
                        frame=frames.HeliographicStonyhurst(obstime=self.epoch))

    def test_visible_and_hidden_roundtrip(self):
        for lon in (0, 45, 180):
            source = source_coordinate(dict(hgs_lon_deg=lon, hgs_lat_deg=12, assumed_radius_km=695700), self.epoch)
            for obs in (self.observer(0), self.observer(80, 6), self.observer(-150, -3)):
                with self.subTest(lon=lon, observer=obs.lon):
                    result = transform_with_check(source, obs)
                    self.assertLess(result["roundtrip_3d_error_m"], 0.01)
                    self.assertGreater(result["observer_source_distance_m"], 0)

    def test_center_projection(self):
        source = source_coordinate(dict(hgs_lon_deg=0, hgs_lat_deg=0, assumed_radius_km=695700), self.epoch)
        result = transform_with_check(source, self.observer(0))
        self.assertTrue(result["surface_visible"])
        self.assertAlmostEqual(result["tx_arcsec"], 0, places=7)
        self.assertAlmostEqual(result["ty_arcsec"], 0, places=7)

    def test_hidden_2d_inverse_is_ambiguous(self):
        source = source_coordinate(dict(hgs_lon_deg=180, hgs_lat_deg=0, assumed_radius_km=695700), self.epoch)
        obs = self.observer(0)
        frame = frames.Helioprojective(observer=obs, obstime=self.epoch)
        target = source.transform_to(frame)
        direction_only = SkyCoord(target.Tx, target.Ty, frame=frame)
        recovered = direction_only.make_3d().transform_to(source.frame)
        separation = np.linalg.norm((recovered.cartesian.xyz-source.cartesian.xyz).to_value(u.km))
        self.assertGreater(separation, 1e6)
        self.assertFalse(transform_with_check(source, obs)["surface_visible"])


if __name__ == "__main__":
    unittest.main()
