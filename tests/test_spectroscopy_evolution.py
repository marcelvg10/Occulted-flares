"""Scientific bookkeeping checks for phase-dependent spectral comparisons."""
import unittest
from unittest.mock import patch
import numpy as np
from occulted_flares.spectroscopy.evolution import supported_upper,fit_sequence


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.edges=np.array([[6,8],[8,28],[28,36],[36,40],[40,45],[45,50],[50,70]],float)

    def test_support_requires_each_tail_channel_and_both_backgrounds(self):
        a=dict(rate=np.full(7,10.),sigma=np.ones(7))
        b=dict(rate=np.full(7,10.),sigma=np.ones(7))
        b['sigma'][5]=3
        upper=supported_upper(self.edges,{'early':a,'late':b},candidates=[70,50,45,40,36,28])
        self.assertEqual(upper,45.)
        b['rate'][2]=-10
        self.assertEqual(supported_upper(self.edges,{'early':a,'late':b},candidates=[70,45,28]),28.)
        self.assertEqual(b['rate'][2],-10.)

    def test_floor_is_fallback_not_detection_and_edges_must_exist(self):
        d=dict(rate=np.full(7,-1.),sigma=np.ones(7))
        self.assertEqual(supported_upper(self.edges,{'a':d},candidates=[70,28]),28.)
        with self.assertRaises(ValueError):supported_upper(self.edges,{'a':d},candidates=[69,28])

    def test_line_channels_can_be_retained_without_deciding_hard_tail_support(self):
        d=dict(rate=np.full(7,10.),sigma=np.ones(7));d['sigma'][2]=20
        self.assertEqual(supported_upper(self.edges,{'a':d},candidates=[70,28],tail_start=36),70.)
        self.assertEqual(d['sigma'][2],20.)

    def test_range_matching_and_independent_fits_in_each_phase(self):
        settings=dict(models=['1T+PL','2T+PL'],display_background='early',
            display_error_policy='diag',common_range_keV=[6,28],lower_keV=6,
            temperature_bounds_MK=[8,80],photon_index_bounds=[1.5,8])
        intervals=[dict(id='A',upper_keV=70,data={('early','diag'):object()}),
                   dict(id='B',upper_keV=40,data={('early','diag'):object()})]
        with patch('occulted_flares.spectroscopy.components.fit_components',return_value={}) as fit:
            fit_sequence(None,intervals,settings)
            self.assertEqual([c.kwargs['energy_range'] for c in fit.call_args_list],[[6,70],[6,70],[6,40],[6,40]])
            self.assertTrue(all('starts' not in c.kwargs for c in fit.call_args_list))
            fit.reset_mock();fit_sequence(None,intervals,settings,common=True)
            self.assertTrue(all(c.kwargs['energy_range']==[6,28] for c in fit.call_args_list))
