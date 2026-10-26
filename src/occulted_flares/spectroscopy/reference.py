"""Conditional first isothermal fit through a checked custom sunkit-spex interface.

No downloads, IDL execution or output writes. Corrected pseudo-counts use chi2,
never a raw-Poisson likelihood. Atomic data are explicit local CHIANTI v7.1.
"""
import contextlib
import importlib.metadata
import os
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ATOMIC_FILES={'chianti_continuum':'chianti_cont_1_250_v71.sav',
              'chianti_lines':'chianti_lines_1_10_v71.sav',
              'xray_abundances':'xray_abun_file.genx'}
_thermal_function=None


def initialize_backend(work):
    """Load pinned legacy API with local manager overrides BEFORE thermal import.

    Relative filenames avoid SunPy's Windows file-URI drive parsing issue.
    Existing global thermal grids in this process are explicitly reset locally.
    """
    global _thermal_function
    if importlib.metadata.version('sunkit-spex')!='0.5.0':
        raise RuntimeError('Reference implementation tested with sunkit-spex==0.5.0')
    from occulted_flares.geometry.geometry_runtime import configure_cache
    configure_cache()
    from sunpy.data import manager
    directory=Path(work)/'ssw/packages/xray/dbase/chianti'
    with contextlib.ExitStack() as stack:
        for key,name in ATOMIC_FILES.items():
            p=directory/name
            if not p.is_file():raise FileNotFoundError(p)
            stack.enter_context(manager.override_file(key,os.path.relpath(p)))
        from sunkit_spex.legacy.fitting.fitter import Fitter
        from sunkit_spex.legacy.fitting.photon_models_for_fitting import f_vth
        from sunkit_spex.legacy import thermal
        thermal.CONTINUUM_GRID=thermal.setup_continuum_parameters()
        thermal.LINE_GRID=thermal.setup_line_parameters()
        thermal.DEFAULT_ABUNDANCES=thermal.setup_default_abundances()
    _thermal_function=f_vth
    return Fitter


def reference_thermal(T,EM,energies=None):
    """T in MK, EM in 1e46 cm^-3; photon density at 1 AU, coronal_ext abundance.

    v7.1 continuum ends at ~200 keV. Bins above its domain are explicitly zero;
    fit bounds T<=80 MK make this distant tail negligible, audited separately.
    This cutoff is not applied to an eventual high-energy continuum model.
    """
    if _thermal_function is None:raise RuntimeError('Call initialize_backend(work) first')
    e=np.asarray(energies,float)
    if (e.ndim!=2 or e.shape[1]!=2 or np.any(e[:,1]<=e[:,0])
            or not np.isfinite(e).all() or T<=0 or EM<0):
        raise ValueError('Positive thermal parameters and finite energy-bin pairs required')
    valid=e[:,1]<=200.15819869050395
    result=np.zeros(len(e))
    if valid.any():result[valid]=_thermal_function(T,EM,energies=e[valid])
    return result


def channel_mask(edges,energy_range):
    e=np.asarray(edges,float);lo,hi=energy_range
    m=(e[:,0]>=lo)&(e[:,1]<=hi)
    if (m.sum()<=2 or e[m][0,0]!=lo or e[m][-1,1]!=hi):
        raise ValueError('Fit range must enclose complete channels and leave >2 channels')
    return m


def fit_data(product,source,background,error_policy='stored_only'):
    """Create explicit signed net rate/error with background mean variance once.

    SYS sensitivity adds file fractions times integrated OBSERVED SOURCE rate,
    once after time integration, as diagonal weights only, not a total budget.
    """
    from .bridge import subtract_candidate
    net=subtract_candidate(source,background)
    sigma=net['sigma'].copy()
    if error_policy=='file_sys_diagonal':
        fraction=np.asarray(product['systematic'][0],float)
        sigma=np.hypot(sigma,fraction*np.abs(source['rate']))
    elif error_policy!='stored_only':raise ValueError('Unknown error policy')
    if not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Finite positive supplied errors required')
    return dict(rate=net['rate'],sigma=sigma,live_s=source['live_s'],
        source_rate=source['rate'],background_rate=background['rate'],error_policy=error_policy)


def load_reference(work,event):
    """Read retained inputs and whole native-bin candidates; no acceptance."""
    from .preparation import load_setup,file_sha256
    from .bridge import read_products,integrate_rows
    from occulted_flares.instrument.products import read_product
    from occulted_flares.lightcurves.background import complete_interval
    work,event=Path(work),Path(event);stage=event/'Spectroscopy'
    settings=json.loads((stage/'reference_fit.json').read_text(encoding='utf-8'))
    meta=json.loads((stage/'outputs/bridge_metadata.json').read_text(encoding='utf-8'))
    for name,record in meta['products'].items():
        if file_sha256(stage/'outputs'/name)!=record['sha256']:raise ValueError('Bridge product changed: '+name)
    product=read_products(stage/'outputs/cal20_prepared_spectrum.fits',stage/'outputs/cal20_response.fits')
    cpd=read_product(load_setup(work,event)['paths']['cpd'])
    draft=json.loads((event/'Lightcurves/feature_review.json').read_text(encoding='utf-8'))
    proposal=next(p for p in draft['proposals'] if p['id']==settings['interval_id'])
    ids=np.flatnonzero(complete_interval(cpd,proposal['start'],proposal['end']))
    source=integrate_rows(product,ids)
    signal=json.loads((event/'Lightcurves/signal.json').read_text(encoding='utf-8'))
    backgrounds={};windows=[]
    for candidate in signal['background_candidates']:
        name=candidate['name']
        if name not in settings['background_alternatives']:continue
        ii=np.flatnonzero(complete_interval(cpd,candidate['start'],candidate['end']))
        backgrounds[name]=integrate_rows(product,ii)
        windows.append(dict(name=name,start_UTC=cpd['start'][ii[0]].isot,end_UTC=cpd['end'][ii[-1]].isot,native_bins=len(ii)))
    data={(name,policy):fit_data(product,source,b,policy) for name,b in backgrounds.items()
          for policy in settings['error_policies']}
    return dict(work=work,event=event,stage=stage,settings=settings,product=product,cpd=cpd,
        source=source,backgrounds=backgrounds,windows=windows,data=data,
        native_start=cpd['start'][ids[0]],native_end=cpd['end'][ids[-1]])


def custom_dictionary(product,data):
    """Public CustomLoader interface with every field explicit; no sign clipping."""
    edges=product['channels'];pe=product['photon_edges'];width=np.diff(edges).ravel()
    live=data['live_s']
    return dict(photon_channel_bins=pe,photon_channel_mids=pe.mean(axis=1),
        photon_channel_binning=np.diff(pe).ravel(),count_channel_bins=edges,
        count_channel_mids=edges.mean(axis=1),count_channel_binning=width,
        counts=data['rate']*live,count_error=data['sigma']*live,
        count_rate=data['rate']/width,count_rate_error=data['sigma']/width,
        effective_exposure=live,srm=product['response'],
        extras={'counts=data-bg':True,'pseudo_counts':True,'error_policy':data['error_policy']})


def predict(product,temperature,em46=1.):
    pe=product['photon_edges']
    return (reference_thermal(temperature,em46,energies=pe)*np.diff(pe).ravel())@product['response']


def s1_thermal(theta_MK,emission_measure46,energies=None):
    """Self-contained registered wrapper; uses the public string-model API."""
    from occulted_flares.spectroscopy.reference import reference_thermal
    return reference_thermal(theta_MK,emission_measure46,energies=energies)


def fit_reference(Fitter,product,data,energy_range,start=(30.,100.),bounds=((8.,80.),(1e-4,1e6))):
    """Actual sunkit-spex fit; also return explicit count-space chi2/residuals."""
    bounds=tuple(tuple(map(float,b)) for b in bounds)
    mask=channel_mask(product['channels'],energy_range)
    fitter=Fitter(custom_dictionary(product,data))
    fitter.add_photon_model(s1_thermal,overwrite=True)
    fitter.model='s1_thermal'
    fitter.loglikelihood='chi2'
    fitter.energy_fitting_range=list(energy_range)
    names=list(fitter.params.param_name)
    for name,value,bound in zip(names,start,bounds):
        fitter.params[name]={'Value':value,'Bounds':bound,'Status':'free'}
    fitter.fit(method='Nelder-Mead',options={'maxiter':700,'xatol':1e-6,'fatol':1e-6})
    values=np.asarray(fitter.params.param_value,float)
    model=predict(product,*values)
    residual=(data['rate']-model)/data['sigma']
    result=dict(T_MK=float(values[0]),EM_46=float(values[1]),model=model,residual=residual,
        chi2=float(np.sum(residual[mask]**2)),dof=int(mask.sum()-2),mask=mask,
        success=bool(fitter._minimize_solution.success),message=str(fitter._minimize_solution.message),
        range_keV=list(energy_range),error_policy=data['error_policy'])
    # v0.5.0 legacy chi2 returns -sum(r**2), then minimizes -2*that.
    # Its internal objective is 2*conventional chi2; the optimum is unchanged.
    np.testing.assert_allclose(2*result['chi2'],fitter._minimize_solution.fun,rtol=2e-7,atol=1e-6)
    result['backend_objective_over_chi2']=float(fitter._minimize_solution.fun/result['chi2'])
    result['at_bound']=bool(any(np.isclose(v,b[0],rtol=1e-4) or np.isclose(v,b[1],rtol=1e-4) for v,b in zip(values,bounds)))
    return result,fitter


def thermal_profile(product,data,energy_range,temperatures):
    """Exact weighted linear EM at each T; independent optimizer check only.

    Delta-chi2 cannot be interpreted as coverage if the model/errors fail.
    """
    mask=channel_mask(product['channels'],energy_range)
    y=data['rate'][mask];sigma=data['sigma'][mask]
    rows=[]
    for temperature in temperatures:
        template=predict(product,float(temperature))[mask]
        em=max(0.,float(np.sum(template*y/sigma**2)/np.sum(template**2/sigma**2)))
        rows.append([float(temperature),em,float(np.sum(((y-em*template)/sigma)**2))])
    return np.asarray(rows)


def profile_minimum(product,data,energy_range,bounds=(8.,80.)):
    """Grid-bracketed one-dimensional check independent of Fitter's two-D search."""
    grid=np.linspace(*bounds,37)
    table=thermal_profile(product,data,energy_range,grid)
    k=int(np.argmin(table[:,2]))
    lower=grid[max(0,k-1)];upper=grid[min(len(grid)-1,k+1)]
    result=minimize_scalar(lambda t:thermal_profile(product,data,energy_range,[t])[0,2],
        bounds=(lower,upper),method='bounded',options={'xatol':1e-7})
    answer=thermal_profile(product,data,energy_range,[result.x])[0]
    return dict(T_MK=float(answer[0]),EM_46=float(answer[1]),chi2=float(answer[2]),success=bool(result.success))


def ospex_control(context,case_name):
    """Read actual OSPEX fit; recompute common chi2 rather than use its divisor."""
    from scipy.io import readsav
    background='late_post' if case_name.startswith('late_post') else 'early_pre'
    policy='file_sys_diagonal' if 'file_sys_diagonal' in case_name else 'stored_only'
    data=context['data'][(background,policy)]
    p=context['product'];directory=context['stage']/'outputs'
    saved=readsav(directory/f'ospex_reference_{case_name}.sav',python_dict=True)
    rates=np.asarray(saved['rates'],float).ravel();errors=np.asarray(saved['errors'],float).ravel()
    np.testing.assert_allclose(rates,data['rate'],rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(errors,data['sigma'],rtol=2e-6,atol=1e-8)
    summary=saved['summary']
    if not np.all(summary['spex_summ_fit_done'][0]):raise ValueError('OSPEX fit not completed')
    mask=np.asarray(summary['spex_summ_emask'][0]).ravel().astype(bool)
    np.testing.assert_array_equal(mask,channel_mask(p['channels'],saved['energy_range'].ravel()))
    params=np.asarray(saved['parameters'],float).ravel()
    model=np.asarray(saved['model'],float).ravel()
    residual=(rates-model)/errors
    chi2=float(np.sum(residual[mask]**2));dof=int(mask.sum()-2)
    native_reduced=float(np.asarray(summary['spex_summ_chisq'][0]).ravel()[0])
    return dict(T_MK=float(params[1]/.08617),EM_46=float(params[0]*1000),
        model=model,residual=residual,chi2=chi2,dof=dof,
        shared_reduced_chi2=chi2/dof,OSPEX_reported_reduced_chi2=native_reduced,
        implied_OSPEX_divisor=chi2/native_reduced,
        stop_message=np.asarray(summary['spex_summ_stop_msg'][0]).ravel()[0].decode(),
        source=saved)


def thermal_tail_sensitivity(product,temperature=80.):
    """Diagnostic exponential extension beyond the atomic table, not new physics.

    Anchor to the last supported bin with an exp(-E/kT)/E tail; compare folding
    to the explicitly zeroed tail. This is a numerical sensitivity, not a bound
    on atomic-model accuracy or a prescription for nonthermal components.
    """
    pe=product['photon_edges'];mid=pe.mean(axis=1)
    density=reference_thermal(temperature,1.,energies=pe)
    valid=pe[:,1]<=200.15819869050395
    j=np.flatnonzero(valid)[-1];extra=np.zeros(len(pe))
    extra[~valid]=density[j]*(mid[j]/mid[~valid])*np.exp(-(mid[~valid]-mid[j])/(temperature*.08617333262))
    model=(density*np.diff(pe).ravel())@product['response']
    change=(extra*np.diff(pe).ravel())@product['response']
    return np.divide(change,model,out=np.zeros_like(change),where=model>0)
