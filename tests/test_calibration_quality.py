import unittest
from pathlib import Path
import tempfile
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from occulted_flares.instrument.calibration import channel_edges, boundary_summary, compression_budget
from occulted_flares.instrument.quality import contiguous_runs, readiness_table, per_bin_quality


class CalibrationQualityTests(unittest.TestCase):
    def test_edge_mask_not_row_number_selects_full_calibration_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask=np.zeros(33,dtype=np.uint8);mask[[1,2,4]]=1
            control=fits.BinTableHDU.from_columns([fits.Column(name='energy_bin_edge_mask',format='33B',array=[mask])],name='CONTROL')
            energy=fits.BinTableHDU.from_columns([
                fits.Column(name='e_low',format='D',array=[4.,5.]),
                fits.Column(name='e_high',format='D',array=[5.,7.])],name='ENERGIES')
            path=Path(tmp)/'sample.fits'
            fits.HDUList([fits.PrimaryHDU(),control,energy]).writeto(path)
            values=np.broadcast_to(np.arange(33.),(32,12,33))+.1
            result=channel_edges(path,dict(energy_edges_keV=values))
            np.testing.assert_array_equal(result['indices'],[1,2,4])
            np.testing.assert_allclose(result['actual'][0,0],[1.1,2.1,4.1])
            summary=boundary_summary(result,[0],[0],[4.])
            self.assertAlmostEqual(summary.iloc[0].actual_min_keV,1.1)

    def test_compression_variance_is_not_an_added_count_rate(self):
        counts=np.array([[[[0.]]],[[[100.]]]])
        comp=np.array([[[[0.]]],[[[300.]]]])
        budget=compression_budget(dict(counts=counts,compression_variance=comp),[0])
        self.assertTrue(np.isnan(budget['raw_relative_sigma'][0,0]))
        self.assertAlmostEqual(budget['compression_variance_fraction'][1,0],.75)
        self.assertAlmostEqual(budget['raw_relative_sigma'][1,0],.2)

    def test_quality_runs_do_not_join_across_an_unselected_bin(self):
        t=Time('2026-07-07')
        cpd=dict(time=t+np.arange(4)*u.s,start=t+np.arange(4)*u.s,
                 end=t+np.arange(1,5)*u.s,duration_s=np.ones(4))
        runs=contiguous_runs(cpd,[True,True,False,True])
        self.assertEqual(runs.bins.tolist(),[2,1])
        self.assertTrue(contiguous_runs(cpd,[False]*4).empty)

    def test_failed_uncertainties_cannot_pass_formal_analysis(self):
        checks=dict(native_counts=True,configuration=True,live_domain=True,
                    calibration_initial=True,background=True,uncertainties=False,response=True)
        table=readiness_table(checks).set_index('analysis')
        self.assertTrue(table.loc['Exploratory central count-rate profiles','required_checks_satisfied'])
        self.assertFalse(table.loc['Physical spectral fitting','required_checks_satisfied'])
        self.assertIn('uncertainties',table.loc['Physical spectral fitting','unresolved_requirements'])

    def test_formal_model_domain_is_not_positive_exposure(self):
        t=Time('2026-07-07')
        cpd=dict(time=Time([t]),start=Time([t]),end=Time([t])+1*u.s,duration_s=np.ones(1))
        model=dict(fraction=np.array([[0.]]),valid=np.array([[True]]))
        budget=dict(counts=np.ones((1,1)))
        ref=dict(detectors=[dict(trigger_group=1)])
        table=per_bin_quality(cpd,model,budget,[0],ref)
        self.assertTrue(table.iloc[0].livetime_domain_valid)
        self.assertFalse(table.iloc[0].positive_finite_live_exposure)


if __name__=='__main__':unittest.main()
