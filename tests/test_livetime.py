import unittest
from pathlib import Path
import numpy as np
from occulted_flares.instrument.detectors import load_reference, detector_groups

from occulted_flares.instrument.livetime import (
    livetime_fraction, map_groups_to_detectors, integrated_detector_rates, compare_quantity)


class LivetimeTests(unittest.TestCase):
    params=dict(tau_s=10.1e-6,eta_s=1.1e-6,beta=.9405910326086957)

    def test_comparison_preserves_failure_and_rejects_nonfinite_agreement(self):
        result=compare_quantity('sigma',[.001,.006],[.001,.001])
        self.assertFalse(result['agrees'])
        self.assertEqual(result['mismatched_elements'],1)
        self.assertFalse(compare_quantity('bad',[np.nan],[np.nan])['agrees'])
        self.assertTrue(compare_quantity('fraction',[.76],[.76])['agrees'])
        with self.assertRaises(ValueError):
            compare_quantity('shape',[1,2],[[1,2]])

    def test_zero_rate_and_saturation_are_not_silently_clipped(self):
        d=np.array([[0,100000]],float)
        out=livetime_fraction(d,np.sqrt(d),[1.],**self.params)
        self.assertEqual(out['fraction'][0,0],1)
        self.assertFalse(out['valid'][0,1])
        self.assertTrue(np.isnan(out['fraction'][0,1]))
        with self.assertRaises(ValueError):
            livetime_fraction(d,np.sqrt(d),[0.],**self.params)

    def test_analytical_fraction_and_uncertainty_derivative(self):
        r=20000.;t=.3;d=np.array([[r*t]])
        out=livetime_fraction(d,[[10.]], [t],**self.params)
        a=self.params['tau_s']+self.params['eta_s'];b=self.params['beta']*self.params['eta_s']
        q=1-a*r
        expected=q*np.exp(-b*r/q)
        self.assertAlmostEqual(out['fraction'][0,0],expected)
        f=lambda rate:(1-a*rate)*np.exp(-b*rate/(1-a*rate))
        derivative=(f(r+.01)-f(r-.01))/.02
        self.assertAlmostEqual(out['fraction_sigma'][0,0],abs(derivative)*10/t,places=10)

    def test_mapping_keeps_nonadjacent_pair(self):
        groups=np.arange(16)[None,:]
        ref=dict(detectors=[dict(trigger_group=k) for k in [3,1,3]])
        np.testing.assert_array_equal(map_groups_to_detectors(groups,ref),[[2,0,2]])

    def test_reference_uses_current_mapping_and_standard_groups(self):
        root=Path(__file__).resolve().parents[1]
        ref=load_reference(root/'config/stix_instrument_reference.json')
        # Published old help examples in stx_ltpair_assignment differ from the
        # executable table updated in 2020. Check the actual source snapshot.
        self.assertEqual(ref['detectors'][0]['trigger_group'],1)
        self.assertEqual(ref['detectors'][10]['trigger_group'],3)
        groups=detector_groups(ref)
        self.assertEqual(groups['cfl'],[8])
        self.assertEqual(groups['bkg'],[9])
        self.assertEqual([len(groups[k]) for k in ['top24','imaging30','fine6']],[24,30,6])
        self.assertEqual(set(groups['top24'])|set(groups['fine6']),set(groups['imaging30']))

    def test_integrated_rates_use_each_detectors_own_exposure(self):
        # Two unequal native durations/exposures, two differently live detectors.
        reduced=dict(counts=np.array([[[[2.]],[[3.]]],[[[8.]],[[9.]]]]))
        live=np.array([[1.,1.],[4.,2.]])
        result=integrated_detector_rates(reduced,live,[True,True],[0,1],0)
        self.assertAlmostEqual(result,10/5+12/3)
        live[0,0]=np.nan
        with self.assertRaises(ValueError):
            integrated_detector_rates(reduced,live,[True,True],[0,1],0)


if __name__=='__main__': unittest.main()
