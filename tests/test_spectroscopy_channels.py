import unittest
import numpy as np
from astropy.time import Time,TimeDelta
from occulted_flares.spectroscopy.channels import (
    summarize_channels, band_indices, boundary_variants, split_native)
from occulted_flares.lightcurves.excess import estimate_excess


class SpectroscopyChannelTests(unittest.TestCase):
    def test_detector_exposure_background_and_signed_precision(self):
        data=dict(counts=np.array([[[20.,0.],[40.,0.]]]),
            compression_variance=np.zeros((1,2,2)),live_s=np.array([[1.,2.]]))
        models={name:dict(rate=np.array([[b,1.],[b,1.]]),variance=np.ones((2,2)),
            zero_count_cells=np.array([[False,True],[False,False]])) for name,b in [('a',2.),('b',3.)]}
        m=dict(data=data,results={name:estimate_excess(**data,model=v) for name,v in models.items()})
        r=summarize_channels(m,[(4,5),(5,6)],models,['a','b'])
        self.assertEqual(r.raw_counts[0],60)
        self.assertEqual(r.expected_background_counts_min[0],6)
        self.assertEqual(r.expected_background_counts_max[0],9)
        self.assertEqual(r.excess_rate_min[0],34)
        self.assertAlmostEqual(r.conditional_sigma_max[0],np.sqrt(32))
        self.assertFalse(r.positive_both[1])
        self.assertTrue(np.isnan(r.worst_conditional_relative_error[1]))
        self.assertEqual(r.maximum_zero_count_background_detector_cells[1],1)

    def test_complete_channel_band_required(self):
        np.testing.assert_array_equal(band_indices([(4,5),(5,6),(6,8)],4,6),[0,1])
        with self.assertRaises(ValueError):band_indices([(4,5),(5,6)],4.5,6)
        with self.assertRaises(ValueError):band_indices([(4,5),(6,8)],4,8)

    def test_boundary_design_and_translation(self):
        v=boundary_variants('2026-07-07T22:52:00','2026-07-07T22:52:25')
        self.assertEqual(len(v),11)
        self.assertAlmostEqual((Time(v[-1]['start'])-Time(v[0]['start'])).sec,3,places=8)
        self.assertAlmostEqual((Time(v[-1]['end'])-Time(v[-1]['start'])).sec,25,places=8)

    def test_whole_bin_split_conserves_ids(self):
        t=Time('2026-07-07T22:52:00')+TimeDelta(np.arange(10)*.3,format='sec')
        p=dict(start=t,end=t+TimeDelta(.3,format='sec'))
        g=split_native(p,np.arange(10))
        np.testing.assert_array_equal(np.concatenate([r['indices'] for r in g]),np.arange(10))
        self.assertEqual([len(r['indices']) for r in g],[4,3,3])
        with self.assertRaises(ValueError):split_native(p,np.array([1,3,5]))
