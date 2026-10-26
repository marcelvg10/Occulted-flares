"""Analytical checks for coverage arithmetic, not implementation snapshots."""
import unittest
import numpy as np
from occulted_flares.geometry.interval_comparison import merge_intervals, covered_seconds, coverage_envelopes


class CoverageTests(unittest.TestCase):
    def test_union_does_not_double_count(self):
        union=merge_intervals([0,1,6],[2,4,8])
        self.assertEqual(covered_seconds(1,7,union),4.)
        np.testing.assert_array_equal(union,[[0,4],[6,8]])

    def test_touching_edges_are_not_overlap(self):
        union=merge_intervals([0,2],[1,3])
        self.assertEqual(covered_seconds(1,2,union),0.)
        self.assertEqual(covered_seconds(2,2,union),0.)

    def test_conservative_bounds_enclose_shifted_windows(self):
        # Bins [0,1],[1,2] translate by any offset in [-0.1,+0.1].
        inner=merge_intervals([.1,1.1],[.9,1.9])
        outer=merge_intervals([-.1,.9],[1.1,2.1])
        # Exposure [0.8,1.3] translates independently in [-0.2,+0.2].
        lower=covered_seconds(1.,1.1,inner)
        upper=covered_seconds(.6,1.5,outer)
        for dx in np.linspace(-.1,.1,11):
            union=merge_intervals([dx,1+dx],[1+dx,2+dx])
            for dy in np.linspace(-.2,.2,11):
                value=covered_seconds(.8+dy,1.3+dy,union)
                self.assertGreaterEqual(value+1e-12,lower)
                self.assertLessEqual(value-1e-12,upper)

    def test_empty_and_invalid(self):
        self.assertEqual(covered_seconds(0,1,merge_intervals([],[])),0.)
        with self.assertRaises(ValueError): merge_intervals([np.nan],[1])

    def test_shared_edge_scenarios_do_not_invent_gaps(self):
        # Common shifts wider than a short bin still preserve continuous coverage.
        inner, outer=coverage_envelopes([0,.3,.6],[.3,.6,.9],
            [-.4,-.1,.2],[.4,.7,1.],[-.1,.2,.5],[.7,1.,1.3])
        np.testing.assert_allclose(inner,[[.4,.5]])
        np.testing.assert_allclose(outer,[[-.4,1.3]])
        # A genuine native gap cannot be bridged by grouping neighbouring bins.
        inner,_=coverage_envelopes([0,2],[1,3],[-.1,1.9],[.1,2.1],[.9,2.9],[1.1,3.1])
        self.assertEqual(covered_seconds(1.2,1.8,inner),0.)


if __name__ == '__main__':
    unittest.main()
