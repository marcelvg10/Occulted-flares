"""Count-space component budgets and provisional imaging handoff helpers.

No component identity, imaging feasibility or physical acceptance follows
automatically from these summaries. Native arrival times remain unchanged.
"""
import numpy as np
from .components import MODELS


def whole_band(channels,low,high):
    """Exact finite contiguous measured channels; even one channel is allowed."""
    e=np.asarray(channels,float)
    if e.ndim!=2 or e.shape[1]!=2 or not np.isfinite(e).all() or high<=low:
        raise ValueError('Finite complete measured energy band required')
    m=(e[:,0]>=low)&(e[:,1]<=high)
    if (not m.any() or e[m][0,0]!=low or e[m][-1,1]!=high
            or np.any(e[m][1:,0]!=e[m][:-1,1])):
        raise ValueError('Band must exactly match contiguous whole channels')
    return m


def component_budget(product,result,bands):
    """Integrated folded count fractions; never extrapolate a component claim.

    Sum predicted channel COUNT rates, not per-keV densities or unweighted
    per-channel fractions. Values outside the actual fitted mask are undefined.
    These are fractions assigned by a model, not separated physical sources.
    """
    nthermal,has_pl=MODELS[result['model_name']];answer=[]
    for low,high in bands:
        m=whole_band(product['channels'],low,high)
        inside=bool(np.all(result['mask'][m]))
        total=float(result['model'][m].sum()) if inside else np.nan
        thermal=float(result['components'][m,:nthermal].sum()) if inside else np.nan
        continuum=float(result['components'][m,-1].sum()) if inside and has_pl else (0. if inside else np.nan)
        valid=inside and total>0
        if inside:np.testing.assert_allclose(thermal+continuum,total,rtol=1e-10,atol=1e-12)
        answer.append(dict(low_keV=low,high_keV=high,band=f'{low:g}-{high:g}',
            inside_fit=inside,predicted_rate=total,
            thermal_fraction=thermal/total if valid else np.nan,
            PL_fraction=continuum/total if valid else np.nan))
    return answer


def band_diagnostics(product,source,background,bands):
    """Signed net band rate and conditional diagonal error; no image S/N.

    Rates are calibrated detector count rates, not incident photon flux or
    raw imaging counts. Per-channel SYS terms enter once, at the source mean.
    Cross-channel covariance is NOT inferred by summing diagonal variances.
    """
    if np.intersect1d(source['rows'],background['rows']).size:
        raise ValueError('Disjoint source and quiet rows required')
    systematic=np.asarray(product['systematic'][0])*np.abs(source['rate'])
    answer=[]
    for low,high in bands:
        m=whole_band(product['channels'],low,high)
        net=float((source['rate'][m]-background['rate'][m]).sum())
        variance=float((source['variance'][m]+background['variance'][m]+systematic[m]**2).sum())
        sigma=np.sqrt(variance)
        answer.append(dict(low_keV=low,high_keV=high,band=f'{low:g}-{high:g}',
            net_rate=net,sigma_diagonal=float(sigma),
            conditional_relative_error=float(sigma/net) if net>0 else np.nan,
            positive_net=bool(net>0)))
    return answer


def imaging_trial(item,band,purpose):
    """In-memory draft with exact native edges; no acceptance or time shift."""
    return dict(interval_id=item['id'],low_keV=float(band[0]),high_keV=float(band[1]),
        first_native_row=int(item['rows'][0]),last_native_row_inclusive=int(item['rows'][-1]),
        start_UTC=item['start'].isot,end_UTC=item['end'].isot,
        native_bins=len(item['rows']),mean_live_s=float(item['source']['live_s']),
        time_convention='Native Solar Orbiter arrival UTC',purpose=purpose,
        status='Draft imaging trial; visibility quality and reconstruction pending',accepted=False)
