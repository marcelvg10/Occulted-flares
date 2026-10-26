"""Nominal-channel diagnostics before daily calibration/response fitting.

These reuse the conditional fixed-live-exposure count/background budget.
They do not select a final fitting mask or assign detection probabilities.
"""
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from occulted_flares.instrument.detectors import reduce_pixels
from occulted_flares.lightcurves.excess import prepare_candidates
from occulted_flares.lightcurves.review import interval_count_budget


def prepare_channels(context):
    energy=context['cpd']['energies']
    low,high=energy.e_low.to_numpy(),energy.e_high.to_numpy()
    selected=np.isfinite(low)&np.isfinite(high)&(high>low)
    bands=list(zip(low[selected],high[selected]))
    reduced=reduce_pixels(context['cpd']['path'],bands)
    prepared=prepare_candidates(dict(context,reduced=reduced))
    return bands,prepared


def band_indices(bands,low,high):
    edges=np.asarray(bands,float)
    ids=np.flatnonzero((edges[:,0]>=low)&(edges[:,1]<=high))
    if (not len(ids) or edges[ids[0],0]!=low or edges[ids[-1],1]!=high
            or np.any(edges[ids[1:],0]!=edges[ids[:-1],1])):
        raise ValueError('Diagnostic band must match complete contiguous channels')
    return ids


def summarize_channels(measurement,bands,models,names):
    """Central candidate spread and conditional precision, without acceptance."""
    if not names: raise ValueError('At least one background alternative required')
    edges=np.asarray(bands,float)
    results=[measurement['results'][name] for name in names]
    net=np.stack([r['excess_rate'][0] for r in results])
    sigma=np.stack([r['conditional_sigma'][0] for r in results])
    if edges.shape!=(net.shape[1],2) or np.any(~np.isfinite(edges)) or np.any(edges[:,1]<=edges[:,0]):
        raise ValueError('Finite increasing channel bounds must match measurement')
    positive=(net>0).all(axis=0)
    relative=np.divide(sigma,net,out=np.full_like(net,np.nan),where=net>0)
    worst=np.where(positive,relative.max(axis=0),np.nan)
    spread=np.divide(np.ptp(net,axis=0),net.min(axis=0),
        out=np.full(net.shape[1],np.nan),where=positive)
    budget=interval_count_budget(measurement,models,names)
    zero=np.stack([models[name]['zero_count_cells'].sum(axis=0) for name in names]).max(axis=0)
    return pd.DataFrame(dict(low_keV=edges[:,0],high_keV=edges[:,1],
        raw_counts=budget['raw_counts'],
        expected_background_counts_min=budget['background_counts_min'],
        expected_background_counts_max=budget['background_counts_max'],
        excess_rate_min=net.min(axis=0),excess_rate_max=net.max(axis=0),
        conditional_sigma_max=sigma.max(axis=0),positive_both=positive,
        worst_conditional_relative_error=worst,relative_background_spread=spread,
        maximum_zero_count_background_detector_cells=zero))


def band_values(measurement,bands,low,high,names):
    """Return candidate central integrated rates, summing complete channels."""
    ids=band_indices(bands,low,high)
    return np.array([measurement['results'][name]['excess_rate'][0,ids].sum() for name in names])


def boundary_variants(start,end,native_s=.3,coarse_s=3.):
    """Manual sensitivity design; variants overlap and are not independent."""
    if not np.isfinite([native_s,coarse_s]).all() or min(native_s,coarse_s)<=0:
        raise ValueError('Positive finite shifts required')
    start,end=Time(start),Time(end)
    if not start<end: raise ValueError('Interval end must follow start')
    offsets=[('baseline',0.,0.)]
    for step in [native_s,coarse_s]:
        offsets.extend([(f'start {sign*step:+g}s',sign*step,0.) for sign in [-1,1]])
        offsets.extend([(f'end {sign*step:+g}s',0.,sign*step) for sign in [-1,1]])
    offsets.extend([(f'translate {sign*coarse_s:+g}s',sign*coarse_s,sign*coarse_s) for sign in [-1,1]])
    rows=[]
    for label,a,b in offsets:
        lo,hi=start+TimeDelta(a,format='sec'),end+TimeDelta(b,format='sec')
        if not lo<hi: raise ValueError('A sensitivity shift reverses the interval')
        rows.append(dict(variant=label,start=lo.isot,end=hi.isot,start_shift_s=a,end_shift_s=b))
    return rows


def split_native(product,indices,parts=3):
    """Split consecutive whole native bins, conserving every selected bin.

    Parts have near-equal numbers of bins, not necessarily equal duration.
    Returned requests use a 1 ns outward numerical tolerance on actual edges.
    """
    ids=np.asarray(indices)
    if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)<parts
            or not isinstance(parts,int) or parts<2 or np.any(np.diff(ids)!=1)
            or np.any(ids<0) or np.any(ids>=len(product['start']))):
        raise ValueError('Enough consecutive valid native bins and parts>=2 required')
    tolerance=TimeDelta(1e-9,format='sec')
    return [dict(indices=g,start=product['start'][g[0]]-tolerance,
                 end=product['end'][g[-1]]+tolerance)
            for g in np.array_split(ids,parts)]
