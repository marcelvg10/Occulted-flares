import unittest
import numpy as np
from occulted_flares.lightcurves.review import (
    balanced_halves, signed_normalization, interval_contrast, interval_count_budget)


class FeatureReviewTests(unittest.TestCase):
    def test_halves_partition_resolution_order(self):
        ref=dict(detectors=[dict(label=x) for x in ['4a','3a','4b','3b']])
        halves=balanced_halves(ref,[0,1,2,3])
        np.testing.assert_equal(halves['top24_half_A'],[1,0])
        np.testing.assert_equal(halves['top24_half_B'],[3,2])
        self.assertEqual(set(halves['top24_half_A']) | set(halves['top24_half_B']),set(range(4)))

    def test_signed_scale_preserves_negative(self):
        x=signed_normalization([[-1,0],[2,0]], [True,True])
        np.testing.assert_equal(x[:,0],[-.5,1])
        self.assertTrue(np.isnan(x[:,1]).all())

    def test_shared_background_cancels_from_contrast(self):
        def m(counts, indices):
            return dict(indices=np.array(indices),data=dict(counts=np.array([[[counts]]],float),
                compression_variance=np.zeros((1,1,1)),live_s=np.ones((1,1))))
        first,second=m(20,[0]),m(10,[1])
        model=dict(rate=np.array([[5.]]),variance=np.array([[10000.]]))
        r=interval_contrast(first,second,model)
        np.testing.assert_equal(r['difference_rate'],[10.])
        np.testing.assert_allclose(r['conditional_count_sigma'],np.sqrt([30.]))
        with self.assertRaises(ValueError):interval_contrast(first,first,model)

    def test_count_budget_exposure_and_model_range(self):
        m=dict(data=dict(counts=np.array([[[10.],[30.]]]),live_s=np.array([[1.,2.]])))
        models=dict(a=dict(rate=np.array([[2.],[3.]])),b=dict(rate=np.array([[3.],[4.]])))
        b=interval_count_budget(m,models,['a','b'])
        np.testing.assert_equal(b['raw_counts'],[40.])
        np.testing.assert_equal(b['background_counts_min'],[8.])
        np.testing.assert_equal(b['background_counts_max'],[11.])
        np.testing.assert_equal(b['largest_detector_raw_count_fraction'],[.75])
