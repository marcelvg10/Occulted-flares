import unittest
import numpy as np
from occulted_flares.spectroscopy.components import fit_components
from occulted_flares.spectroscopy.sensitivity import (
    shared_background_covariance,systematic_covariance,shifted_rows)


class SensitivityTests(unittest.TestCase):
    class Toy:
        product=dict(channels=np.column_stack([np.arange(6,70,8),np.arange(14,78,8)]))
        def templates(self,model,parameters):
            return np.exp(-self.product['channels'].mean(axis=1)/parameters['T1'])[:,None]

    def data(self):
        toy=self.Toy();y=3*np.exp(-toy.product['channels'].mean(axis=1)/25)
        return toy,dict(rate=y+np.array([.01,-.01,.02,0,-.01,0,.01,-.01]),sigma=np.full(8,.04))

    def test_diagonal_covariance_preserves_existing_fit(self):
        toy,d=self.data();a=fit_components(toy,d,'1T')
        b=fit_components(toy,{**d,'covariance':np.diag(d['sigma']**2)},'1T')
        np.testing.assert_allclose(a['model'],b['model'],rtol=1e-6)
        self.assertAlmostEqual(a['chi2'],b['chi2'],places=8)

    def test_masked_correlated_objective_uses_marginal_covariance(self):
        toy,d=self.data();C=.04**2*(.6*np.eye(8)+.4*np.ones((8,8)))
        r=fit_components(toy,{**d,'covariance':C},'1T',energy_range=(6,54))
        m=r['mask'];diff=d['rate'][m]-r['model'][m]
        expected=diff@np.linalg.solve(C[np.ix_(m,m)],diff)
        self.assertAlmostEqual(r['chi2'],expected,places=8)
        self.assertAlmostEqual(r['thermal_parameters'][0]['T_MK'],25,delta=2)

    def test_invalid_covariance_is_rejected(self):
        toy,d=self.data()
        for C in [np.ones((8,8))*.04**2,np.eye(8),np.full((8,7),.01)]:
            with self.assertRaises(ValueError):fit_components(toy,{**d,'covariance':C},'1T')

    def test_same_marginal_errors_do_not_determine_correlations(self):
        _,d=self.data();s=np.linspace(.01,.04,8)
        a=systematic_covariance(d,s,0);b=systematic_covariance(d,s,.95)
        np.testing.assert_allclose(a['sigma'],b['sigma'])
        self.assertGreater(b['covariance'][0,1],0)
        np.linalg.cholesky(b['covariance'])

    def test_shared_background_cancels_in_rate_difference(self):
        C=shared_background_covariance([4,9],25)
        self.assertEqual(np.array([1,-1])@C@np.array([1,-1]),13)
        self.assertEqual(C[0,1],25)

    def test_native_row_shifts_preserve_width_and_reject_outside_data(self):
        item=dict(rows=np.arange(10,20));r=shifted_rows(item,3,30)
        np.testing.assert_array_equal(r,np.arange(13,23))
        with self.assertRaises(ValueError):shifted_rows(item,-11,30)
        with self.assertRaises(ValueError):shifted_rows(item,.5,30)
