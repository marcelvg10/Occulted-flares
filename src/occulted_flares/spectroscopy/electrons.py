"""Isotropic cold-target electron hypotheses through the retained STIX DRM.

The adapter reuses the tested positive-amplitude count fitter. Its internal
gamma key carries the ELECTRON flux index delta; exposed results rename it.
No photon-index conversion is used. Target amplitudes have different units.
"""
from functools import lru_cache
import numpy as np
from .components import CountModels, fit_components


def electron_shape(energy, delta, cutoff, high=1000.):
    """Unit-integral electron flux power law between finite cutoffs [1/keV]."""
    energy=np.asarray(energy,float)
    if not np.isfinite([delta,cutoff,high]).all() or delta<=2 or not 0<cutoff<high:
        raise ValueError('delta > 2 and positive ordered cutoffs required')
    norm=(delta-1)/(cutoff**(1-delta)-high**(1-delta))
    return np.where((energy>=cutoff)&(energy<=high),norm*np.maximum(energy,cutoff)**(-delta),0.)


def mean_energy(delta, cutoff, high=1000.):
    """Finite power-law mean kinetic energy, keV; same flux convention."""
    electron_shape([cutoff],delta,cutoff,high)
    return ((delta-1)/(delta-2)*(cutoff**(2-delta)-high**(2-delta))
            /(cutoff**(1-delta)-high**(1-delta)))


def electron_photons(edges,target,delta,cutoff,high=1000.,order=4):
    """Bin-average photons at 1 AU per OSPEX-scale amplitude.

    thin: n V integral(Fbar dE) / 1e55 [cm^-2 s^-1]; efd=True.
    thick: injected integral(Fdot dE) / 1e35 [electrons s^-1].
    Equal slopes below/above an internal break remove a fitted electron break.
    Gauss-Legendre integration resolves the cutoff inside a photon bin.
    """
    from sunkit_spex.legacy.emission import bremsstrahlung_thin_target,bremsstrahlung_thick_target
    e=np.asarray(edges,float)
    electron_shape([cutoff],delta,cutoff,high)
    if (target not in ('thin','thick') or e.ndim!=2 or e.shape[1]!=2
            or not np.isfinite(e).all() or np.any(e<=0) or np.any(e[:,1]<=e[:,0]) or order<2):
        raise ValueError('Valid target, positive photon bins and quadrature order >= 2 required')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    # Split only bins that straddle the electron cutoff (a curvature change).
    answer=np.zeros(len(e));brk=np.sqrt(cutoff*high)
    function=bremsstrahlung_thin_target if target=='thin' else bremsstrahlung_thick_target
    for lo,hi in [(e[:,0],np.minimum(e[:,1],cutoff)),(np.maximum(e[:,0],cutoff),e[:,1])]:
        valid=hi>lo
        if not valid.any():continue
        points=((lo[valid]+hi[valid])[:,None]+(hi[valid]-lo[valid])[:,None]*nodes)/2
        kwargs={'efd':True} if target=='thin' else {}
        values=function(points.ravel(),delta,brk,delta,cutoff,high,**kwargs).reshape(points.shape)
        answer[valid]+=((values@weights)*(hi[valid]-lo[valid])/2)/np.diff(e[valid]).ravel()
    answer*=1e55 if target=='thin' else 1e35
    if not np.isfinite(answer).all() or np.any(answer<0):raise ArithmeticError('Invalid bremsstrahlung spectrum')
    return answer


class ElectronModels(CountModels):
    def __init__(self,product,target,cutoff,high=1000.,order=4,thermal_engine=None):
        super().__init__(product)
        if thermal_engine is not None:self.thermal=thermal_engine.thermal
        self.target,self.cutoff,self.high,self.order=target,float(cutoff),float(high),order
        self.nonthermal=lru_cache(maxsize=1024)(self._fold)

    def _fold(self,delta):
        e=self.product['photon_edges']
        return (electron_photons(e,self.target,delta,self.cutoff,self.high,self.order)
                *np.diff(e).ravel())@self.product['response']

    def templates(self,model,parameters):
        n=2 if model=='2T+PL' else 1
        return np.column_stack([self.thermal(parameters[f'T{i+1}']) for i in range(n)]
                               +[self.nonthermal(float(parameters['gamma']))])


def fit_electrons(engine,data,nthermal=2,energy_range=(6,70)):
    """Fit T, delta and positive amplitudes at a FIXED electron cutoff.

    Use the count fitter's gamma coordinate internally, then relabel it.
    Nominal dof is conditional on the fixed cutoff; a grid minimum is not a CI.
    """
    if nthermal not in (1,2):raise ValueError('One or two thermal components required')
    starts=[dict(T1=t,T2=h,gamma=d) for t,h,d in
            [(10.,28.,4.),(20.,38.,5.),(10.,50.,6.),(25.,65.,3.)]]
    r=fit_components(engine,data,f'{nthermal}T+PL',energy_range=energy_range,
        photon_index_bounds=(2.05,9.),starts=starts,max_nfev=160)
    # The base fitter's best trial shares its parameter dictionary with the
    # trial ledger. Copy it before relabelling either representation.
    r['parameters']=dict(r['parameters'])
    r['electron_index']=r['parameters'].pop('gamma')
    r['electron_norm']=float(r['amplitudes'][-1])
    r['photon_index']=None;r['photon_norm_at_30']=None
    r['model_name']=f'{nthermal}T+{engine.target}'
    r['target']=engine.target;r['cutoff_keV']=engine.cutoff;r['high_keV']=engine.high
    r['at_nonlinear_bound']=['delta' if k=='gamma' else k for k in r['at_nonlinear_bound']]
    for trial in r['trial_solutions']:
        trial['parameters']['delta']=trial['parameters'].pop('gamma')
    r['parameters']['delta']=r['electron_index']
    r['injected_power_erg_s']=(r['electron_norm']*1e35*mean_energy(r['electron_index'],engine.cutoff,engine.high)
                               *1.602176634e-9 if engine.target=='thick' else None)
    return r
