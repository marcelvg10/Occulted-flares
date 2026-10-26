"""Independent analytical and convention checks for finite photon travel times."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from occulted_flares.geometry.photon_timing import SourceModels, solve_emission, solve_arrival, emission_envelope, recover_native, read_stix_intervals, C_KM_S
from occulted_flares.geometry.timing_ephemeris import Trajectory


class LinearBody:
    def __init__(self, x=0., velocity=0.):
        self.x, self.velocity = x, velocity
    def position(self, seconds):
        seconds = np.asarray(seconds)
        return np.array([self.x+self.velocity*seconds, np.zeros_like(seconds), np.zeros_like(seconds)])


class TimingTest(unittest.TestCase):
    def source(self, sun=None):
        return SourceModels(np.array([[1.],[0.],[0.]]), np.array([0.]), np.array([0.]), 0.,
                            np.array([0.,0.,1.]), sun or LinearBody(), rotation_rad_s=0.)

    def test_stationary_exact_and_inverse(self):
        obs = LinearBody(C_KM_S*400)
        reception = np.array([1000.,1200.])
        result = solve_emission(reception, obs, self.source())
        np.testing.assert_allclose(result[0], reception-400, atol=1e-9, rtol=0)
        np.testing.assert_allclose(solve_arrival(result[0], obs, self.source())[0], reception, atol=1e-9, rtol=0)

    def test_moving_source_and_receiver_analytical(self):
        source_speed, observer_speed, distance = 12., 30., C_KM_S*400
        reception = np.array([1000.,1500.])
        expected = ((C_KM_S-observer_speed)*reception-distance)/(C_KM_S-source_speed)
        actual = solve_emission(reception, LinearBody(distance, observer_speed), self.source(LinearBody(0,source_speed)))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)

    def test_shift_reversal_preserves_leap_second(self):
        native = Time(["2016-12-31T23:59:60", "2017-01-01T00:00:00"], scale="utc")
        shifted = native+142.5*u.s
        recovered, history = recover_native(shifted, "earth_equivalent_fixed", 142.5)
        np.testing.assert_allclose((recovered-native).to_value(u.s), 0, atol=1e-9)
        self.assertTrue(history)
        with self.assertRaises(ValueError):
            recover_native(shifted, "unknown")
        with self.assertRaises(ValueError):
            recover_native(shifted, "earth_equivalent_fixed")

    def test_centiseconds_are_bin_centres_with_finite_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"synthetic.fits"
            h = fits.Header({"LEVEL":"L1", "DATE-BEG":"2026-07-07T22:00:00", "TIMESYS":"UTC"})
            data = fits.BinTableHDU.from_columns([fits.Column(name="time", format="J", unit="cs", array=[15,45]),
                                                 fits.Column(name="timedel", format="J", unit="cs", array=[30,30])], name="DATA")
            fits.HDUList([fits.PrimaryHDU(header=h),data]).writeto(path)
            result = read_stix_intervals(path)
            t0 = Time(h["DATE-BEG"],scale="utc")
            np.testing.assert_allclose((result["start"]-t0).to_value(u.s), [0.,.3], atol=1e-9)
            np.testing.assert_allclose((result["end"]-t0).to_value(u.s), [.3,.6], atol=1e-9)

    def test_scenario_envelope_at_every_supplied_edge(self):
        model = SourceModels(np.array([[1.,1.],[0.,0.],[0.,0.]]), np.array([0.,0.]), np.array([0.,1000.]),
                             695700., np.array([0.,0.,1.]), LinearBody(), rotation_rad_s=0.)
        observer = LinearBody(C_KM_S*400)
        edges = np.array([1000.,1000.3,1001.,2000.])
        lo, hi = emission_envelope(edges, observer, model, chunk_size=2)
        np.testing.assert_allclose(lo, edges-400+695700/C_KM_S, atol=1e-9, rtol=0)
        np.testing.assert_allclose(hi, edges-400+696700/C_KM_S, atol=1e-9, rtol=0)

    def test_no_trajectory_extrapolation(self):
        trajectory = Trajectory([0.,60.], [[0.,0.,0.],[60.,0.,0.]], [[1.,0.,0.],[1.,0.,0.]])
        self.assertAlmostEqual(trajectory.position(30.)[0],30.)
        with self.assertRaises(ValueError):
            trajectory.position(-0.1)


if __name__ == "__main__":
    unittest.main()
