import unittest
import numpy as np
from occulted_flares.spectroscopy.reference import channel_mask,custom_dictionary,fit_data


class ReferenceInterfaceTests(unittest.TestCase):
    def test_complete_fit_channel_range_required(self):
        edges=np.array([[4,5],[5,6],[6,8],[8,10]])
        np.testing.assert_array_equal(channel_mask(edges,[5,10]),[False,True,True,True])
        with self.assertRaises(ValueError):channel_mask(edges,[5.5,10])
        with self.assertRaises(ValueError):channel_mask(edges,[6,10])

    def test_custom_interface_keeps_signs_and_normalization(self):
        p=dict(channels=np.array([[4,5],[5,7]]),photon_edges=np.array([[3,4],[4,6]]),response=np.array([[1,2],[3,4]]))
        d=dict(rate=np.array([-2.,4.]),sigma=np.array([3.,5.]),live_s=10.,error_policy='stored_only')
        result=custom_dictionary(p,d)
        np.testing.assert_array_equal(result['counts'],[-20,40])
        np.testing.assert_array_equal(result['count_rate'],[-2,2])
        np.testing.assert_array_equal(result['count_error'],[30,50])
        np.testing.assert_array_equal(result['srm'],p['response'])
        self.assertTrue(result['extras']['counts=data-bg'])

    def test_systematic_sensitivity_after_integration_once(self):
        p=dict(systematic=np.array([[.05]]))
        source=dict(rate=np.array([10.]),variance=np.array([1.]),rows=np.array([0,1]),live_s=20.)
        background=dict(rate=np.array([2.]),variance=np.array([4.]),rows=np.array([2,3]),live_s=30.)
        d=fit_data(p,source,background,'file_sys_diagonal')
        np.testing.assert_array_equal(d['rate'],[8])
        np.testing.assert_allclose(d['sigma'],np.sqrt([5+.25]))
        with self.assertRaises(ValueError):fit_data(p,source,background,'silent_default')
