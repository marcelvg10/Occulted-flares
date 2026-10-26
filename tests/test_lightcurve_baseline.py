import unittest
import numpy as np
from occulted_flares.lightcurves.baseline import (
    group_profiles, normalized_profiles, positive_ratio, selection_summary)


class BaselineTests(unittest.TestCase):
    def test_correct_per_detector_and_partition(self):
        c = np.array([[[[10.]], [[20.]], [[30.]]]])
        p = group_profiles(c, [2.], [[1., 2., .5]],
                           dict(top=[0, 1], fine=[2], imaging=[0, 1, 2]))
        np.testing.assert_equal(p['top']['live_rate'], [[20.]])
        np.testing.assert_equal(p['imaging']['live_rate'],
                                p['top']['live_rate']+p['fine']['live_rate'])
        self.assertEqual(p['imaging']['observed_rate'][0, 0], 30.)

    def test_bad_exposure_and_duplicate_selection_fail(self):
        c = np.ones((2, 3, 1, 1))
        with self.assertRaises(ValueError):
            group_profiles(c, [1, 1], [[1, 0, 1], [1, 1, 1]], dict(chosen=[1]))
        with self.assertRaises(ValueError):
            group_profiles(c, [1, 1], np.ones((2, 3)), dict(chosen=[1, 1]))

    def test_normalization_uses_common_window_and_handles_zero(self):
        p, peak = normalized_profiles([[100, 0], [2, 0], [4, 0]], [False, True, True])
        np.testing.assert_equal(peak, [4, 0])
        np.testing.assert_equal(p[:, 0], [25, .5, 1])
        self.assertTrue(np.isnan(p[:, 1]).all())
        self.assertTrue(np.isnan(positive_ratio([0, 1], [0, 2])[0]))

    def test_summary_weights_native_duration(self):
        c = np.array([[[[2.]]], [[[8.]]]])
        p = group_profiles(c, [1, 4], [[.5], [4]], dict(g=[0]))
        row = selection_summary(p, [1, 4], [True, True], [[4, 12]]).iloc[0]
        self.assertEqual(row.raw_counts, 10)
        self.assertEqual(row.observed_mean_counts_s, 2)
        self.assertAlmostEqual(row.time_weighted_live_rate_counts_s, 2.4)
        # Integrated live-weighted mean 10/4.5 is deliberately different.
        self.assertNotAlmostEqual(row.time_weighted_live_rate_counts_s, 10/4.5)
