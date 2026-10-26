import unittest
import numpy as np
from occulted_flares.spectroscopy.components import (
    powerlaw_average,redistribution,positive_amplitudes,fit_components)


class ComponentMathTests(unittest.TestCase):
    def test_photon_bin_integral_and_index_one_limit(self):
        np.testing.assert_allclose(powerlaw_average([[10,20]],2,pivot_keV=10),[.5])
        for gamma in [1.,1.+1e-12]:
            np.testing.assert_allclose(powerlaw_average([[10,20]],gamma,pivot_keV=10),[np.log(2)],rtol=1e-10)
        with self.assertRaises(ValueError):powerlaw_average([[5,4]],3)

    def test_shift_identity_and_no_wrap_or_count_creation(self):
        edges=np.array([[4,5],[5,6],[6,7]])
        np.testing.assert_array_equal(redistribution(edges,0),np.eye(3))
        np.testing.assert_array_equal(redistribution(edges,1),[[0,0,0],[1,0,0],[0,1,0]])
        np.testing.assert_array_equal(redistribution(edges,.5),[[.5,0,0],[.5,.5,0],[0,.5,.5]])
        self.assertTrue(np.all(redistribution(edges,-.4).sum(axis=0)<=1))

    def test_linear_amplitudes_and_signed_data(self):
        design=np.array([[1.,0],[1,1],[0,2]])
        truth=np.array([2.,3.]);data=design@truth
        np.testing.assert_allclose(positive_amplitudes(design,data,np.array([1.,2.,3.])),truth)
        signed=np.array([-3.,2.]);copy=signed.copy()
        np.testing.assert_allclose(positive_amplitudes(np.eye(2),signed,np.ones(2)),[0,2])
        np.testing.assert_array_equal(signed,copy)

    def test_identical_components_do_not_imply_unique_amplitudes(self):
        design=np.ones((4,2));amplitudes=positive_amplitudes(design,np.full(4,3.),np.ones(4))
        self.assertAlmostEqual(amplitudes.sum(),3.)
        self.assertTrue(np.all(amplitudes>=0))

    def test_temperature_amplitude_injection_recovery(self):
        class Toy:
            product=dict(channels=np.column_stack([np.arange(6,70,8),np.arange(14,78,8)]))
            def templates(self,model,parameters):
                return np.exp(-self.product['channels'].mean(axis=1)/parameters['T1'])[:,None]
        toy=Toy();data=dict(rate=3*np.exp(-toy.product['channels'].mean(axis=1)/25),sigma=np.full(8,.01))
        result=fit_components(toy,data,'1T',max_nfev=80)
        self.assertTrue(result['success']);self.assertAlmostEqual(result['thermal_parameters'][0]['T_MK'],25,places=4)
        self.assertAlmostEqual(result['thermal_parameters'][0]['EM_46'],3,places=4)
        self.assertEqual(result['dof'],6);self.assertLess(result['chi2'],1e-10)
