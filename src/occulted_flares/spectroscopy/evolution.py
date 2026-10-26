"""Whole-bin spectral evolution and explicit, provisional energy support.

No acceptance or exports. Shared background means are reused, not independent
measurements in successive intervals. Range selection is descriptive/data-led.
"""
import json
import numpy as np
from .bridge import integrate_rows
from .reference import fit_data
from occulted_flares.lightcurves.background import complete_interval


def load_intervals(context, identifiers):
    """Resolve draft phase windows to nonoverlapping whole native rows.

    Timing must match the retained calibrated product. Gaps between draft
    phases are preserved. No interval boundaries are inferred from fits.
    """
    draft=json.loads((context['event']/'Lightcurves/feature_review.json').read_text(encoding='utf-8'))
    proposals={p['id']:p for p in draft['proposals']}
    if not identifiers or len(set(identifiers))!=len(identifiers):
        raise ValueError('Unique nonempty candidate IDs required')
    product,cpd=context['product'],context['cpd']
    for edge in ('start','end'):
        if len(product[edge])!=len(cpd[edge]) or np.max(np.abs((product[edge]-cpd[edge]).sec))>1e-5:
            raise ValueError('Calibrated and native arrival edges do not match')
    answer=[];used=set()
    for name in identifiers:
        p=proposals[name]
        if p['category']!='spectral':raise ValueError('Expected spectral draft intervals')
        ids=np.flatnonzero(complete_interval(cpd,p['start'],p['end']))
        if not len(ids) or used.intersection(ids):raise ValueError('Empty or overlapping draft intervals')
        used.update(ids)
        source=integrate_rows(product,ids)
        answer.append(dict(id=name,purpose=p['purpose'],rows=ids,source=source,
            start=cpd['start'][ids[0]],end=cpd['end'][ids[-1]],
            requested_start=p['start'],requested_end=p['end']))
    return answer


def supported_upper(channels, alternatives, candidates=(70,63,56,50,45,40,36,28),
                    floor=28., relative_error_goal=.25,tail_start=None):
    """Largest complete upper range with supported tail under ALL backgrounds.

    Each channel from tail_start up to upper must be positive with sigma/net
    <= the illustrative goal. Default tail_start is floor. An explicit higher
    start can keep calibration-line channels from deciding hard-tail support;
    those channels remain in the FIT and need separate review. Fallback floor
    is NOT evidence of adequate counts below it. No selection uses residuals.
    """
    e=np.asarray(channels,float)
    if e.ndim!=2 or e.shape[1]!=2 or np.any(e[:,1]<=e[:,0]) or not alternatives:
        raise ValueError('Channel pairs and background alternatives required')
    if not 0<relative_error_goal<1 or not np.any(e[:,1]==floor):
        raise ValueError('Positive fractional goal and complete floor edge required')
    tail_start=floor if tail_start is None else float(tail_start)
    if tail_start<floor or not np.any(e[:,0]==tail_start):raise ValueError('Complete tail start >= floor required')
    for upper in sorted(set(candidates),reverse=True):
        if upper<floor or not np.any(e[:,1]==upper):raise ValueError('Complete upper edges >= floor required')
        mask=(e[:,0]>=tail_start)&(e[:,1]<=upper)
        if upper==floor:return float(floor)
        if not mask.any():continue
        good=True
        for data in alternatives.values():
            y,s=np.asarray(data['rate']),np.asarray(data['sigma'])
            if y.shape!=(len(e),) or s.shape!=y.shape:raise ValueError('One rate/error per channel required')
            good &= bool(np.all(np.isfinite(y[mask])&np.isfinite(s[mask])&(s[mask]>0)
                                 &(y[mask]>0)&(s[mask]<=relative_error_goal*y[mask])))
        if good:return float(upper)
    raise ValueError('Include the floor in candidate edges for explicit fallback')


def prepare_evolution(context,settings):
    """Integrations, both error policies and declared range-support rule."""
    intervals=load_intervals(context,settings['candidate_ids'])
    product=context['product']
    for item in intervals:
        item['data']={(bg,policy):fit_data(product,item['source'],context['backgrounds'][bg],policy)
            for bg in settings['background_alternatives'] for policy in settings['error_policies']}
        alternatives={bg:item['data'][bg,settings['display_error_policy']]
            for bg in settings['background_alternatives']}
        item['upper_keV']=supported_upper(product['channels'],alternatives,
            candidates=settings['upper_candidates_keV'],floor=settings['tail_support_floor_keV'],
            relative_error_goal=settings['illustrative_relative_error_goal'],tail_start=settings['tail_support_start_keV'])
    return intervals


def fit_sequence(models,intervals,settings,background=None,policy=None,common=False):
    """Independent multistart fits; never propagate a prior phase optimum only.

    Warm starts alone could falsely turn optimizer branch changes into physics.
    Each phase keeps the full notebook-04 starting grid and all local trials.
    """
    from .components import fit_components
    background=background or settings['display_background']
    policy=policy or settings['display_error_policy']
    answer={}
    for item in intervals:
        bounds=settings['common_range_keV'] if common else [settings['lower_keV'],item['upper_keV']]
        for name in settings['models']:
            answer[item['id'],name]=fit_components(models,item['data'][background,policy],name,
                energy_range=bounds,temperature_bounds=settings['temperature_bounds_MK'],
                photon_index_bounds=settings['photon_index_bounds'])
    return answer


def evolution_row(item,result,background,policy,comparison):
    """Displayed diagnostics, not confidence intervals or accepted parameters."""
    thermal=result['thermal_parameters'];two=len(thermal)==2
    return dict(interval=item['id'],model=result['model_name'],comparison=comparison,
        background=background,weights=policy,n_channels=int(result['mask'].sum()),
        T1_MK=thermal[0]['T_MK'],EM1_46=thermal[0]['EM_46'],
        T2_MK=thermal[1]['T_MK'] if two else np.nan,
        EM2_46=thermal[1]['EM_46'] if two else np.nan,
        gamma=result['photon_index'],A30=result['photon_norm_at_30'],
        chi2=result['chi2'],dof=result['dof'],chi2_dof=result['chi2']/result['dof'],
        success=result['success'],bounds=', '.join(result['at_nonlinear_bound']) or 'none',
        weak_component=result['weak_component'])
