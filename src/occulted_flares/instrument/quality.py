"""Separate computational checks from scientific readiness and missing inputs."""
import astropy.units as u
import numpy as np
import pandas as pd


def contiguous_runs(cpd,selection):
    """List real continuous runs of selected bins; never bridge a missing bin."""
    mask=np.asarray(selection,bool)
    if mask.shape!=(len(cpd['time']),):
        raise ValueError('Selection must have one value per native bin')
    indices=np.flatnonzero(mask)
    rows=[]
    if not len(indices): return pd.DataFrame(columns=['first_bin','last_bin','native_start_utc','native_end_utc','bins','integration_s'])
    breaks=np.flatnonzero((np.diff(indices)!=1)|
        (np.abs((cpd['start'][indices[1:]]-cpd['end'][indices[:-1]]).to_value(u.s))>1e-6))+1
    for run in np.split(indices,breaks):
        rows.append(dict(first_bin=int(run[0]),last_bin=int(run[-1]),
            native_start_utc=cpd['start'][run[0]].isot,native_end_utc=cpd['end'][run[-1]].isot,
            bins=len(run),integration_s=float(cpd['duration_s'][run].sum())))
    return pd.DataFrame(rows)


def per_bin_quality(cpd,model,budget,detectors,reference):
    """Measured/sampled diagnostics, not a single automatic 'good science' mask."""
    groups=np.unique([reference['detectors'][i]['trigger_group']-1 for i in detectors])
    values=model['fraction'][:,groups]
    minimum=np.min(values,axis=1)  # Invalid values intentionally remain visible.
    return pd.DataFrame(dict(native_bin=np.arange(len(cpd['time'])),
        native_start_utc=cpd['start'].isot,native_end_utc=cpd['end'].isot,
        integration_s=cpd['duration_s'],
        livetime_domain_valid=np.all(model['valid'][:,groups],axis=1),
        positive_finite_live_exposure=np.all(np.isfinite(values)&(values>0),axis=1),
        minimum_candidate_livetime_fraction=minimum,
        all_diagnostic_band_counts_finite=np.all(np.isfinite(budget['counts']),axis=1)))


def readiness_table(checks):
    """Explicit dependency mapping. Missing/failed checks cannot become a pass."""
    requirements={
        'Exploratory central count-rate profiles':['native_counts','configuration','live_domain'],
        'Initial energy-calibration diagnostics':['calibration_initial'],
        'Quantitative variability/significance':['native_counts','configuration','live_domain','background','uncertainties'],
        'Physical spectral fitting':['calibration_initial','background','uncertainties','response'],
        'Absolute STIX/context registration':['aspect','source_geometry_review'],
        'Absolute subsecond multi-observer timing':['clock','source_geometry_review'],
    }
    rows=[]
    for analysis,needed in requirements.items():
        missing=[key for key in needed if not (
            isinstance(checks.get(key),(bool,np.bool_)) and bool(checks[key]))]
        rows.append(dict(analysis=analysis,required_checks_satisfied=not missing,
                         unresolved_requirements=', '.join(missing) if missing else 'None for this limited diagnostic'))
    return pd.DataFrame(rows)
