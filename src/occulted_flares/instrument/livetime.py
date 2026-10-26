"""Explicit translation of installed GSW livetime equations, not an IDL run.

The code follows the executable stx_livetime_fraction.pro body, not its older
header example. Uncertainty propagation includes Poisson plus telemetry
compression terms, but not parameter errors or count/trigger covariance.
"""
import numpy as np


def compare_quantity(name, actual, expected, rtol=2e-6, atol=1e-8):
    """Report agreement or disagreement without relaxing tolerances to pass.

    A readable failed comparison is a scientific result, not an execution
    success for that quantity. Shape mismatches remain hard errors.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f'Unexpected IDL array shape for {name}: {actual.shape}')
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual-expected)
    agrees = finite & (difference <= atol+rtol*np.abs(expected))
    return dict(quantity=name, agrees=bool(np.all(agrees)),
                mismatched_elements=int(np.count_nonzero(~agrees)),
                maximum_absolute_difference=float(np.max(difference[finite])) if finite.any() else np.nan,
                rtol=rtol, atol=atol)


def livetime_fraction(trigger_counts,trigger_sigma,duration_s,*,tau_s,eta_s,beta):
    d=np.asarray(trigger_counts,float)
    sigma=np.asarray(trigger_sigma,float)
    duration=np.asarray(duration_s,float)
    if (d.ndim!=2 or sigma.shape!=d.shape or duration.shape!=(len(d),)
            or not np.isfinite(d).all() or not np.isfinite(sigma).all()
            or np.any(d<0) or np.any(sigma<0) or not np.isfinite(duration).all()
            or np.any(duration<=0) or not np.isfinite([tau_s,eta_s,beta]).all()
            or tau_s<=0 or eta_s<0 or not 0<=beta<=1):
        raise ValueError('Invalid triggers, durations or model parameters')
    rate=d/duration[:,None]
    q=1-rate*(tau_s+eta_s)
    valid=q>0
    incoming=np.full(d.shape,np.nan)
    np.divide(rate,q,out=incoming,where=valid)
    attenuation=np.exp(-beta*eta_s*incoming)
    fraction=q*attenuation
    fraction[~valid]=np.nan
    derivative=attenuation*((tau_s+eta_s)+beta*eta_s/np.where(valid,q,np.nan))
    fraction_sigma=derivative*sigma/duration[:,None]
    live=fraction*duration[:,None]
    return dict(trigger_rate_s=rate,inferred_input_rate_s=incoming,fraction=fraction,
                fraction_sigma=fraction_sigma,live_s=live,valid=valid,
                live_sigma_s=fraction_sigma*duration[:,None],saturation_margin=q)


def map_groups_to_detectors(values,reference):
    """Map all groups first; do not remove a paired detector from its triggers."""
    array=np.asarray(values)
    if array.ndim!=2 or array.shape[1]!=16:
        raise ValueError('Expected time × 16 trigger groups')
    indices=np.array([d['trigger_group']-1 for d in reference['detectors']])
    return array[:,indices]


def corrected_detector_rates(reduced,live_s,live_sigma_s):
    """Sum pixels then divide each detector separately by its own live exposure.

    Only a diagnostic statistical approximation is returned; trigger/count
    covariance and time-parameter systematics remain unmodelled.
    """
    counts=reduced['counts'].sum(axis=2)
    variance=reduced['variance'].sum(axis=2)
    live=np.asarray(live_s,float)
    sigma=np.asarray(live_sigma_s,float)
    if live.shape!=counts.shape[:2] or sigma.shape!=live.shape:
        raise ValueError('Live exposure must match time × detector')
    good=np.isfinite(live)&(live>0)
    denom=np.where(good,live,np.nan)[...,None]
    rate=counts/denom
    error=np.sqrt(variance/denom**2+(counts*sigma[...,None]/denom**2)**2)
    return dict(rate=rate,approximate_sigma=error)


def integrated_detector_rates(reduced,live_s,selection,detector_indices,band):
    """Live-exposure-weighted rate per detector, then sum detectors.

    Not the mean of per-bin rates, and not total counts divided by one pooled
    live exposure. This matches the GSW integrated pixel-data aggregation; it
    is not a native-time-weighted incident-rate average when rate and livetime
    covary. Reject incomplete model validity instead of dropping bins.
    """
    chosen=np.asarray(selection,bool)
    indices=np.asarray(detector_indices,int)
    if not chosen.any() or not len(indices) or len(np.unique(indices))!=len(indices):
        raise ValueError('Non-empty selection and unique detector indices required')
    live=np.asarray(live_s)[chosen][:,indices]
    if not np.isfinite(live).all() or np.any(live<=0):
        raise ValueError('Selected integration has invalid livetime; no silent omission')
    counts=reduced['counts'][chosen,:,:,band].sum(axis=2)[:,indices]
    return float(np.sum(counts.sum(axis=0)/live.sum(axis=0)))
