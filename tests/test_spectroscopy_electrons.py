import unittest
import numpy as np
from scipy.integrate import quad
from occulted_flares.spectroscopy.electrons import (
    electron_shape, mean_energy, electron_photons, ElectronModels, fit_electrons)


class ElectronTests(unittest.TestCase):
    def test_unit_integral_and_energy_moment(self):
        for d in [2.1,4.,8.]:
            f=lambda e:electron_shape([e],d,12,1000)[0]
            self.assertAlmostEqual(quad(f,12,1000)[0],1.,places=8)
            self.assertAlmostEqual(quad(lambda e:e*f(e),12,1000)[0],mean_energy(d,12,1000),places=7)
        np.testing.assert_array_equal(electron_shape([5,1001],4,12),[0,0])

    def test_installed_distribution_and_amplitude_units(self):
        from sunkit_spex.legacy.emission import (
            BrokenPowerLawElectronDistribution,bremsstrahlung_thin_target,bremsstrahlung_thick_target)
        e=np.geomspace(12,999,100)
        distribution=BrokenPowerLawElectronDistribution(p=4,q=4,eelow=12,eebrk=np.sqrt(12000),eehigh=1000)
        np.testing.assert_allclose(distribution.flux(e),electron_shape(e,4,12),rtol=1e-12)
        # Narrow bins approach direct photon evaluation; checks 1e55/1e35 once.
        bins=np.array([[29.99999,30.00001],[49.99999,50.00001]])
        for target,func,scale in [('thin',bremsstrahlung_thin_target,1e55),('thick',bremsstrahlung_thick_target,1e35)]:
            expected=func(bins.mean(axis=1),4,np.sqrt(12000),4,12,1000)*scale
            np.testing.assert_allclose(electron_photons(bins,target,4,12),expected,rtol=1e-6)

    def test_cutoff_split_quadrature_and_bad_inputs(self):
        bins=np.array([[6,12],[12,20],[20,30],[30,70]],float)
        for target in ['thin','thick']:
            a=electron_photons(bins,target,4,16,order=8)
            b=electron_photons(bins,target,4,16,order=16)
            self.assertTrue(np.all(a>0));np.testing.assert_allclose(a,b,rtol=.001)
        with self.assertRaises(ValueError):electron_photons(bins,'invalid',4,16)
        with self.assertRaises(ValueError):electron_shape([20],2,16)
        with self.assertRaises(ValueError):mean_energy(4,1000,100)

    def test_synthetic_fit_recovers_electron_index_and_not_photon_label(self):
        # Real electron emission and identity response, independent synthetic data.
        edges=np.column_stack([np.arange(6,70,2),np.arange(8,72,2)])
        product=dict(photon_edges=edges,channels=edges,response=np.eye(len(edges)))
        engine=ElectronModels(product,'thick',16)
        engine.thermal=lambda t:10*np.exp(-edges.mean(axis=1)/(t*.08617333))
        templates=engine.templates('1T+PL',dict(T1=25,gamma=4.5))
        rate=templates@np.array([200.,.5])
        r=fit_electrons(engine,dict(rate=rate,sigma=np.maximum(rate*.02,.001)),1)
        self.assertTrue(r['success'])
        self.assertAlmostEqual(r['electron_index'],4.5,places=3)
        self.assertAlmostEqual(r['thermal_parameters'][0]['T_MK'],25.,places=2)
        self.assertAlmostEqual(r['electron_norm'],.5,places=3)
        self.assertIsNone(r['photon_index']);self.assertNotIn('gamma',r['parameters'])
        self.assertGreater(r['injected_power_erg_s'],0)
