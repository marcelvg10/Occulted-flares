import unittest
import numpy as np
from astropy.time import Time
from occulted_flares.spectroscopy.interpretation import (
    whole_band,component_budget,band_diagnostics,imaging_trial)


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        self.product=dict(channels=np.array([[4,6],[6,8],[8,12],[12,20]],float),
            systematic=np.array([[.1,.1,.1,.1]]))

    def test_exact_bands_allow_single_channel_but_reject_partial_or_gaps(self):
        np.testing.assert_array_equal(whole_band(self.product['channels'],6,8),[False,True,False,False])
        with self.assertRaises(ValueError):whole_band(self.product['channels'],7,12)
        with self.assertRaises(ValueError):whole_band([[6,8],[10,12]],6,12)

    def test_folded_fractions_sum_rates_and_do_not_extrapolate(self):
        components=np.array([[1,1],[9,1],[2,8],[1,9]],float)
        r=dict(model_name='1T+PL',components=components,model=components.sum(axis=1),
            mask=np.array([False,True,True,False]))
        rows=component_budget(self.product,r,[(6,12),(12,20)])
        self.assertAlmostEqual(rows[0]['PL_fraction'],9/20)
        self.assertAlmostEqual(rows[0]['thermal_fraction'],11/20)
        self.assertFalse(rows[1]['inside_fit']);self.assertTrue(np.isnan(rows[1]['PL_fraction']))

    def test_signed_net_rate_and_variance_budget_are_preserved(self):
        source=dict(rate=np.array([10,20,30,40.]),variance=np.array([1,4,9,16.]),rows=np.array([1]))
        bg=dict(rate=np.array([11,22,33,44.]),variance=np.ones(4),rows=np.array([0]))
        row=band_diagnostics(self.product,source,bg,[(6,12)])[0]
        self.assertEqual(row['net_rate'],-5.)
        self.assertAlmostEqual(row['sigma_diagonal']**2,4+9+1+1+4+9)
        self.assertFalse(row['positive_net']);self.assertTrue(np.isnan(row['conditional_relative_error']))

    def test_imaging_draft_keeps_exact_edges_and_never_accepts(self):
        item=dict(id='S1',rows=np.array([10,11]),start=Time('2026-07-07T22:52:00.126'),
            end=Time('2026-07-07T22:52:00.726'),source=dict(live_s=.45))
        r=imaging_trial(item,[8,12],'test');self.assertFalse(r['accepted'])
        self.assertEqual(r['start_UTC'],'2026-07-07T22:52:00.126')
        self.assertEqual(r['last_native_row_inclusive'],11)
