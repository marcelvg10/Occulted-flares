import unittest
import numpy as np
from astropy.time import Time, TimeDelta
from occulted_flares.lightcurves.excess import estimate_excess, combine_native, require_disjoint
from occulted_flares.lightcurves.binning import fixed_groups, adaptive_groups, group_table


def product(n, dt=.3):
    t = Time('2026-07-07T22:00:00')
    return dict(start=t+TimeDelta(np.arange(n)*dt, format='sec'),
                end=t+TimeDelta(np.arange(1, n+1)*dt, format='sec'),
                duration_s=np.full(n, dt))


class ExcessBinningTests(unittest.TestCase):
    def test_excess_variance_and_shared_background_floor(self):
        c = np.array([[[10.]], [[10.]]]); v = np.zeros_like(c); l = np.ones((2, 1))
        bg = dict(rate=np.array([[2.]]), variance=np.array([[4.]]))
        native = estimate_excess(c, v, l, bg)
        np.testing.assert_allclose(native['excess_rate'], [[8], [8]])
        merged = combine_native(c, v, l, [np.array([0, 1])])
        result = estimate_excess(**merged, model=bg)
        # Counting variance falls from 10 to 5, while background variance stays 4.
        np.testing.assert_allclose(result['conditional_sigma'], [[3.]])
        np.testing.assert_allclose(result['background_variance'], [4.])

    def test_negative_excess_and_source_background_overlap(self):
        result = estimate_excess(np.array([[[1.]]]), np.zeros((1, 1, 1)), np.ones((1, 1)),
                                 dict(rate=np.array([[2.]]), variance=np.array([[1.]])))
        self.assertEqual(result['excess_rate'][0, 0], -1.)
        self.assertTrue(np.isnan(result['conditional_relative_error'][0, 0]))
        with self.assertRaises(ValueError): require_disjoint(np.array([1]), [False, True])

    def test_fixed_roundoff_phase_tail_and_conservation(self):
        p = product(8)
        groups = fixed_groups(p, np.ones(8, bool), .9)
        self.assertEqual([len(g) for g in groups], [3, 3, 2])
        shifted = fixed_groups(p, np.ones(8, bool), .9, phase_native_bins=1)
        self.assertEqual([len(g) for g in shifted], [1, 3, 3, 1])
        np.testing.assert_array_equal(np.concatenate(shifted), np.arange(8))
        c = np.arange(8, dtype=float).reshape(8, 1, 1)
        merged = combine_native(c, c/10, np.full((8, 1), .2), groups)
        self.assertEqual(merged['counts'].sum(), c.sum())
        self.assertAlmostEqual(merged['live_s'].sum(), 1.6)
        self.assertTrue(group_table(p, groups, .9).shorter_than_target.iloc[-1])

    def test_no_gaps_or_unselected_bins_bridged(self):
        p = product(4)
        with self.assertRaises(ValueError): fixed_groups(p, [True, False, True, True], .9)
        p['start'][2] += TimeDelta(.01, format='sec')
        with self.assertRaises(ValueError): fixed_groups(p, np.ones(4, bool), .9)

    def test_adaptive_cap_failure_and_native_success(self):
        p = product(7, 1.)
        c = np.full((7, 1, 1), 100.); v = np.zeros_like(c); live = np.ones((7, 1))
        bad = dict(rate=np.array([[90.]]), variance=np.array([[10000.]]))
        groups, report = adaptive_groups(p, np.ones(7, bool), c, v, live, dict(bad=bad), 0,
                                         relative_target=.2, maximum_s=3.)
        self.assertEqual([len(g) for g in groups], [3, 3, 1])
        self.assertFalse(report.target_met.any())
        good = dict(rate=np.array([[1.]]), variance=np.array([[0.]]))
        groups, report = adaptive_groups(p, np.ones(7, bool), c, v, live, dict(good=good), 0)
        self.assertEqual(len(groups), 7)
        self.assertTrue(report.target_met.all())

    def test_adaptive_requires_every_background_and_rejects_split(self):
        p = product(3, 1.)
        c = np.full((3, 1, 1), 100.); v = np.zeros_like(c); l = np.ones((3, 1))
        good = dict(rate=np.array([[1.]]), variance=np.array([[0.]]))
        negative = dict(rate=np.array([[101.]]), variance=np.array([[0.]]))
        g, r = adaptive_groups(p, [True]*3, c, v, l, dict(good=good, negative=negative), 0,
                                maximum_s=2.)
        self.assertFalse(r.target_met.any())
        with self.assertRaises(ValueError):
            adaptive_groups(p, [True]*3, c, v, l, dict(good=good), 0, maximum_s=.5)
