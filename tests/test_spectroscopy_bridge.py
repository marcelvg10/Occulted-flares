import unittest
import numpy as np
from occulted_flares.spectroscopy.bridge import effective_response,integrate_rows,subtract_candidate


class SpectroscopyBridgeTests(unittest.TestCase):
    def test_area_and_measured_width_enter_once(self):
        r=effective_response([[1,2],[3,4]],[[4,5],[5,7]],2)
        np.testing.assert_array_equal(r,[[2,8],[6,16]])
        np.testing.assert_array_equal(np.array([10,20])@r,[140,400])
        with self.assertRaises(ValueError):effective_response([[1]],[[5,4]],2)

    def test_pseudo_count_weighting_preserves_negative_values(self):
        p=dict(rate=np.array([[10.,-2.],[30.,-4.]]),error=np.ones((2,2)),live_s=np.array([1.,3.]))
        r=integrate_rows(p,np.array([0,1]))
        np.testing.assert_allclose(r['rate'],[25,-3.5])
        np.testing.assert_allclose(r['variance'],[10/16,10/16])
        with self.assertRaises(ValueError):integrate_rows(p,np.array([0,0]))

    def test_background_uncertainty_does_not_disappear(self):
        a=dict(rate=np.array([2.]),variance=np.array([1.]),rows=np.array([0]))
        b=dict(rate=np.array([3.]),variance=np.array([4.]),rows=np.array([1]))
        r=subtract_candidate(a,b)
        np.testing.assert_equal(r['rate'],[-1])
        np.testing.assert_allclose(r['sigma'],np.sqrt([5]))
        with self.assertRaises(ValueError):subtract_candidate(a,a)
