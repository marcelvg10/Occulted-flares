import unittest
import numpy as np
from astropy.time import Time, TimeDelta
from occulted_flares.lightcurves.background import (
    fit_constant_background, predict_background, complete_interval, overlap_aggregate)


class BackgroundTests(unittest.TestCase):
    def test_window_review_retains_edges_and_conditional_errors(self):
        from occulted_flares.lightcurves.background_review import window_table
        t = Time('2026-07-07T00:00:00')
        p = dict(start=t+TimeDelta([0, 1], format='sec'),
                 end=t+TimeDelta([1, 3], format='sec'),
                 time=t+TimeDelta([.5, 2], format='sec'))
        c = np.array([[[4.]], [[8.]]]); comp = np.array([[[1.]], [[0.]]])
        live = np.array([[.5], [2.]])
        m = fit_constant_background(c, comp, live, [True, True])
        rows = window_table(p, c, comp, live, [True, True], m, [[4, 12]])
        np.testing.assert_allclose(rows.rate_counts_s, [8, 4])
        np.testing.assert_allclose(rows.conditional_rate_sigma_counts_s, [np.sqrt(5)/.5, np.sqrt(8)/2])
        np.testing.assert_allclose(rows.end_offset_s-rows.start_offset_s, [1, 2])
        np.testing.assert_allclose(rows.count_residual,
            (np.array([4, 8])-4.8*np.array([.5, 2]))/np.sqrt(4.8*np.array([.5, 2])+[1, 0]))

    def test_poisson_exposure_fit_and_shared_prediction(self):
        c = np.array([[[2.], [4.]], [[6.], [8.]]])
        live = np.array([[1., 2.], [3., 2.]])
        m = fit_constant_background(c, np.zeros_like(c), live, [True, True])
        np.testing.assert_allclose(m['rate'], [[2.], [3.]])
        np.testing.assert_allclose(m['variance'], [[.5], [.75]])
        p = predict_background(m, live)
        np.testing.assert_allclose(p['expected_counts'], [[8.], [12.]])
        np.testing.assert_allclose(p['summed_rate_variance'], [1.25])

    def test_zero_counts_flagged_and_empty_selection_rejected(self):
        c = np.zeros((2, 1, 1)); live = np.ones((2, 1))
        m = fit_constant_background(c, c, live, [True, True])
        self.assertTrue(m['zero_count_cells'].all())
        with self.assertRaises(ValueError):
            fit_constant_background(c, c, live, [False, False])

    def test_interval_match_preserves_native_edges(self):
        t = Time('2026-07-07T00:00:00')
        p = dict(start=t+TimeDelta([0, 1], format='sec'), end=t+TimeDelta([1, 3], format='sec'))
        c = np.array([[[4.]], [[4.]]]); live = np.array([[.5], [2.]])
        rate, members = overlap_aggregate(p, c, live, [t], [t+TimeDelta(3, format='sec')])
        np.testing.assert_allclose(rate, [[8/2.5]])
        self.assertEqual(len(members[0]), 2)
        with self.assertRaises(ValueError):
            overlap_aggregate(p, c, live, [t+TimeDelta(.5, format='sec')], [p['end'][-1]])
        np.testing.assert_equal(complete_interval(p, p['start'][1], p['end'][1]), [False, True])
