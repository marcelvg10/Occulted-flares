"""Matched count-space thermal/empirical-continuum comparisons.

Temperature/index are nonlinear; positive component amplitudes are solved by
weighted NNLS at each trial. All fits use direct checked atomic-model calls.
No fitting acceptance, exports, downloads or physical electron interpretation.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares,nnls
from scipy.linalg import solve_triangular
from .reference import predict,channel_mask

MODELS={'1T':(1,False),'2T':(2,False),'1T+PL':(1,True),'2T+PL':(2,True)}


def powerlaw_average(edges,index,pivot_keV=30.):
    """Exact bin-average of unit flux density at pivot; no electron cutoff.

    Defined on the FULL incident response grid, including above 200 keV.
    Gamma near one uses a stable logarithmic limit through expm1.
    """
    e=np.asarray(edges,float)
    if e.ndim!=2 or e.shape[1]!=2 or not np.isfinite(e).all() or np.any(e<=0) or np.any(e[:,1]<=e[:,0]):
        raise ValueError('Positive finite incident energy-bin pairs required')
    if not np.isfinite(index) or not np.isfinite(pivot_keV) or pivot_keV<=0:
        raise ValueError('Finite index and positive pivot required')
    lo,hi=e.T;z=1.-index;ratio=np.log(hi/lo)
    integral=np.log(hi/lo) if abs(z)<1e-9 else lo**z*np.expm1(z*ratio)/z
    return pivot_keV**index*integral/(hi-lo)


def redistribution(edges,offset_keV):
    """Conservative coarse post-response energy shift, NOT an ELUT correction.

    Piecewise uniform within original measured channels. Counts falling beyond
    the available channel range are lost explicitly, never wrapped around.
    A positive offset shifts model counts toward higher measured energy.
    """
    e=np.asarray(edges,float)
    if e.ndim!=2 or e.shape[1]!=2 or np.any(e[:,1]<=e[:,0]) or not np.isfinite(e).all() or not np.isfinite(offset_keV):
        raise ValueError('Finite measured bins and offset required')
    overlap=np.maximum(0.,np.minimum(e[:,1,None],e[None,:,1]+offset_keV)
                            -np.maximum(e[:,0,None],e[None,:,0]+offset_keV))
    return overlap/np.diff(e).ravel()[None,:]


def positive_amplitudes(templates,rate,sigma):
    """Weighted nonnegative linear amplitudes with normalized design columns."""
    a=np.asarray(templates,float);y=np.asarray(rate,float);s=np.asarray(sigma,float)
    if (a.ndim!=2 or a.shape[0]!=len(y) or s.shape!=y.shape or not np.isfinite(a).all()
            or np.any(a<0) or not np.isfinite(y).all() or not np.isfinite(s).all() or np.any(s<=0)):
        raise ValueError('Finite nonnegative templates, signed rates and positive errors required')
    weighted=a/s[:,None];scale=np.linalg.norm(weighted,axis=0)
    if np.any(scale<=0):raise ValueError('Each template needs nonzero response in the fitted channels')
    scaled,_=nnls(weighted/scale,y/s,maxiter=1000)
    return scaled/scale


def comparison_mask(product,energy_range=(6,70),exclude=()):
    mask=channel_mask(product['channels'],energy_range)
    edges=product['channels']
    for lo,hi in exclude:
        if not (np.any(edges[:,0]==lo) and np.any(edges[:,1]==hi)):
            raise ValueError('Exclusions must align with whole measured channels')
        mask&=~((edges[:,0]>=lo)&(edges[:,1]<=hi))
    return mask


class CountModels:
    """Small per-product template cache; no interpolation approximation."""
    def __init__(self,product):
        self.product=product
        self.thermal=lru_cache(maxsize=2048)(lambda t:predict(product,float(t),1.))

    def templates(self,model,parameters):
        nthermal,pl=MODELS[model]
        columns=[self.thermal(float(parameters[f'T{i+1}'])) for i in range(nthermal)]
        if pl:
            e=self.product['photon_edges']
            density=powerlaw_average(e,parameters['gamma'])
            columns.append((density*np.diff(e).ravel())@self.product['response'])
        return redistribution(self.product['channels'],parameters.get('offset',0.))@np.column_stack(columns)


def fit_components(models,data,model,energy_range=(6,70),exclude=(),fixed=None,
                   free_offset=False,starts=None,max_nfev=180,
                   temperature_bounds=(8.,80.),photon_index_bounds=(1.5,8.)):
    """Multi-start conditional chi2, positive linear amplitudes profiled exactly.

    Nominal k includes all component parameters even when amplitudes vanish;
    N-k is descriptive only at boundaries or under incomplete covariance.
    Sorting two fitted temperatures resolves label exchange, not identifiability.
    Optional data['covariance'] supplies a full channel covariance. It is
    restricted to fitted channels BEFORE Cholesky whitening. Diagonal
    residuals remain display diagnostics; chi2 then uses whitened residuals.
    """
    fixed={} if fixed is None else dict(fixed)
    nthermal,pl=MODELS[model]
    keys=[f'T{i+1}' for i in range(nthermal)]+(['gamma'] if pl else [])+(['offset'] if free_offset else [])
    bounds={**{f'T{i+1}':tuple(temperature_bounds) for i in range(nthermal)},
            'gamma':tuple(photon_index_bounds),'offset':(-.4,.4)}
    if any(not np.isfinite(b).all() or b[1]<=b[0] for b in bounds.values()):
        raise ValueError('Finite ordered bounds required')
    if any(not np.isfinite(value) or key in bounds and not bounds[key][0]<=value<=bounds[key][1]
           for key,value in fixed.items()):raise ValueError('Fixed parameter outside its supported bounds')
    # A fixed T2 profile describes the HOT component; constrain the free T1 below it.
    if 'T2' in fixed:bounds['T1']=(temperature_bounds[0],float(fixed['T2']))
    if not set(fixed).issubset(set(keys)|{'offset'}):raise ValueError('Unknown fixed parameter')
    variable=[key for key in keys if key not in fixed]
    lower=np.array([bounds[k][0] for k in variable]);upper=np.array([bounds[k][1] for k in variable])
    if starts is None:
        starts=([dict(T1=t,T2=h,gamma=g,offset=0.) for t,h,g in
                 [(18.,38.,3.),(27.,60.,4.5),(38.,78.,6.),(9.,28.,3.3),
                  (10.,40.,3.),(14.,30.,4.),(10.,70.,3.)]]
                if nthermal==2 else [dict(T1=t,gamma=g,offset=0.) for t,g in
                 [(20.,3.),(32.,4.5),(50.,6.),(10.,3.3),(65.,3.)]])
    mask=comparison_mask(models.product,energy_range,exclude)
    k=len(variable)+nthermal+int(pl)
    if mask.sum()<=k:raise ValueError('Too few measured channels for the nominal parameter count')
    whitening=None
    if 'covariance' in data:
        covariance=np.asarray(data['covariance'],float);n=len(mask)
        if (covariance.shape!=(n,n) or not np.isfinite(covariance).all()
                or not np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-12)
                or not np.allclose(np.diag(covariance),np.asarray(data['sigma'])**2,rtol=1e-8,atol=1e-12)):
            raise ValueError('Symmetric finite covariance matching sigma diagonal required')
        selected_cov=covariance[np.ix_(mask,mask)]
        diagonal=np.diag(selected_cov)
        if np.any(diagonal<=0):raise ValueError('Positive covariance diagonal required')
        correlation=selected_cov/np.sqrt(np.outer(diagonal,diagonal))
        if np.linalg.eigvalsh(correlation).min()<=10*np.finfo(float).eps*mask.sum():
            raise ValueError('Positive definite, numerically nonsingular fitted covariance required')
        try:lower_cov=np.linalg.cholesky(selected_cov)
        except np.linalg.LinAlgError as exc:raise ValueError('Positive definite fitted covariance required') from exc
        whitening=solve_triangular(lower_cov,np.eye(mask.sum()),lower=True)
        whitened_rate=whitening@data['rate'][mask]
    def evaluate(x):
        pars={**dict(zip(variable,x)),**fixed}
        a=models.templates(model,pars)
        if whitening is None:
            amp=positive_amplitudes(a[mask],data['rate'][mask],data['sigma'][mask])
        else:
            # Whitened design columns may have either sign; physical templates
            # and fitted component amplitudes are still nonnegative.
            weighted=whitening@a[mask];scale=np.linalg.norm(weighted,axis=0)
            if np.any(scale<=0):raise ValueError('Nonzero fitted templates required')
            amp=nnls(weighted/scale,whitened_rate,maxiter=1000)[0]/scale
        prediction=a@amp
        residual=(data['rate']-prediction)/data['sigma']
        objective=residual[mask] if whitening is None else whitening@(data['rate'][mask]-prediction[mask])
        return pars,a,amp,prediction,residual,objective
    trials=[]
    for start in starts:
        x0=np.clip([start[k] for k in variable],lower+1e-8,upper-1e-8)
        opt=least_squares(lambda x:evaluate(x)[-1],x0,bounds=(lower,upper),
            diff_step=2e-4,x_scale='jac',max_nfev=max_nfev,ftol=2e-8,xtol=2e-8,gtol=2e-8)
        pars,a,amp,prediction,residual,objective=evaluate(opt.x)
        trials.append(dict(parameters=pars,amplitudes=amp,model=prediction,
            components=a*amp[None,:],residual=residual,chi2=float(np.sum(objective**2)),
            success=bool(opt.success),nfev=int(opt.nfev),message=str(opt.message)))
    successful=[r for r in trials if r['success']]
    best=min(successful or trials,key=lambda r:r['chi2']).copy()
    temperatures=[best['parameters'][f'T{i+1}'] for i in range(nthermal)]
    order=np.argsort(temperatures)
    best['thermal_parameters']=[dict(T_MK=float(temperatures[i]),EM_46=float(best['amplitudes'][i])) for i in order]
    best['photon_index']=float(best['parameters']['gamma']) if pl else None
    best['photon_norm_at_30']=float(best['amplitudes'][-1]) if pl else None
    best['model_name']=model;best['mask']=mask;best['nominal_k']=k;best['dof']=int(mask.sum()-k)
    best['trial_chi2']=[r['chi2'] for r in trials]
    best['trial_success']=[r['success'] for r in trials]
    best['trial_solutions']=[dict(parameters=r['parameters'],amplitudes=r['amplitudes'].tolist(),
        chi2=r['chi2'],success=r['success']) for r in trials]
    best['at_nonlinear_bound']=[key for key in variable if
        min(abs(best['parameters'][key]-bounds[key][0]),abs(best['parameters'][key]-bounds[key][1]))<.02]
    fractions=best['components'][mask].sum(axis=0)/max(best['model'][mask].sum(),1e-300)
    best['component_count_fractions']=fractions
    best['weak_component']=bool(np.any(fractions<1e-4))
    return best


def result_summary(result):
    """JSON/table-friendly diagnostic summary; no fitted errors or acceptance."""
    return {k:v for k,v in result.items() if k not in
        ['model','components','residual','mask','amplitudes','component_count_fractions']}
