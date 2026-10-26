"""Energy-boundary and raw uncertainty diagnostics; no implicit recalibration."""
from pathlib import Path

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
from scipy.io import readsav


def channel_edges(path, calibration):
    """Map the CPD edge mask into the 33 physical CAL edges, without guessing.

    The current CPD omits the underflow channel. Its row numbers alone are
    therefore not indices into the full CAL array. The CONTROL edge mask is
    the mapping; grouped/nonconsecutive edges remain explicit.
    """
    with fits.open(path) as hdus:
        masks=np.asarray(hdus['CONTROL'].data['energy_bin_edge_mask'])
        if masks.ndim!=2 or masks.shape[1]!=33 or not np.all(masks==masks[0]):
            raise ValueError('A constant 33-entry energy-edge mask is required')
        selected=np.flatnonzero(masks[0])
        energies=hdus['ENERGIES'].data
        if len(selected)!=len(energies)+1:
            raise ValueError('Energy mask does not describe the CPD channel table')
        lo=np.asarray(energies['e_low'],float)
        hi=np.asarray(energies['e_high'],float)
        if not np.isfinite(lo).all() or not np.all(lo[1:]==hi[:-1]):
            raise ValueError('Inspect missing/noncontiguous nominal channels')
        # The NaN upper bound in the overflow row is an unbounded channel,
        # never a finite-width bin for counts/keV normalization.
        if np.isnan(hi[-1]): hi[-1]=np.inf
        nominal=np.r_[lo,hi[-1]]
        actual=np.asarray(calibration['energy_edges_keV'])[...,selected]
        if not np.all(np.diff(actual,axis=-1)>0):
            raise ValueError('Non-ordered calibrated boundaries')
    return dict(indices=selected,nominal=nominal,actual=actual,
                nominal_low=lo,nominal_high=hi)


def boundary_summary(edges,detectors,pixels,energies):
    """Pixel-population spread, not statistical uncertainty on a source energy."""
    rows=[]
    for energy in energies:
        where=np.flatnonzero(edges['nominal']==energy)
        if len(where)!=1 or not np.isfinite(energy):
            raise ValueError(f'No unique finite nominal boundary at {energy}')
        actual=edges['actual'][np.ix_(detectors,pixels,[where[0]])].ravel()
        delta=actual-energy
        rows.append(dict(nominal_keV=float(energy),pixels=len(actual),
            actual_min_keV=float(actual.min()),actual_median_keV=float(np.median(actual)),
            actual_max_keV=float(actual.max()),offset_min_keV=float(delta.min()),
            offset_max_keV=float(delta.max())))
    return pd.DataFrame(rows)


def compression_budget(reduced,detectors):
    """Separate Poisson and compression variances for detector/pixel sums.

    Does not include background, livetimes, calibration or correlations. Zero
    counts retain undefined relative errors rather than being assigned 0%.
    """
    counts=reduced['counts'][:,detectors].sum(axis=(1,2))
    comp=reduced['compression_variance'][:,detectors].sum(axis=(1,2))
    if not np.isfinite(counts).all() or not np.isfinite(comp).all() or np.any(comp<0) or np.any(counts<0):
        raise ValueError('Invalid raw uncertainty inputs')
    total=counts+comp
    fraction=np.divide(comp,total,out=np.full_like(total,np.nan),where=total>0)
    relative=np.divide(np.sqrt(total),counts,out=np.full_like(total,np.nan),where=counts>0)
    return dict(counts=counts,poisson_variance=counts,compression_variance=comp,
                compression_variance_fraction=fraction,raw_relative_sigma=relative)


def load_calibration_comparison(path,cpd,calibration,reduced,live_s,detectors):
    """Validate optional actual IDL central-count export before displaying it.

    IDL expands requested intervals to whole enclosing native bins. Select its
    recorded interval here, not the 'fully contained' diagnostic choice in 03/04.
    No uncertainty array from the unresolved IDL reader is accepted by this path.
    """
    saved=readsav(Path(path),python_dict=True)
    decode=lambda x:x.decode() if isinstance(x,bytes) else str(x)
    if decode(saved['source_file'])!=cpd['path'].name or decode(saved['calibration_file'])!=calibration['path'].name:
        raise ValueError('IDL calibration comparison uses other source files')
    np.testing.assert_allclose(saved['cal_e_low'],calibration['energy_edges_keV'][...,:-1],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(saved['cal_e_high'],calibration['energy_edges_keV'][...,1:],rtol=1e-10,atol=1e-10)
    bands=np.asarray(saved['energy_ranges'])
    raw=np.asarray(saved['raw_counts']).transpose(0,2,1)
    corrected=np.asarray(saved['calibrated_counts']).transpose(0,2,1)
    intervals=np.asarray(saved['actual_time_mjd'])
    live_saved=np.asarray(saved['live_seconds'])
    if raw.shape!=corrected.shape or raw.shape!=(len(bands),32,12) or live_saved.shape!=(len(bands),32):
        raise ValueError('Unexpected IDL pixel/count/live exposure dimensions')
    if not np.isfinite(raw).all() or not np.isfinite(corrected).all() or np.any(corrected<0):
        raise ValueError('Invalid central IDL calibrated counts')
    rows=[]
    masks=[]
    for k,(low,high) in enumerate(bands):
        matches=[i for i,band in enumerate(reduced['bands']) if list(band)==[low,high]]
        if len(matches)!=1: raise ValueError('Export band not present in configured diagnostics')
        start,end=Time(intervals[k],format='mjd',scale='utc')
        selected=((cpd['start']-start).to_value(u.s)>=-1e-3)&((cpd['end']-end).to_value(u.s)<=1e-3)
        if not selected.any(): raise ValueError('IDL interval has no complete matching native bins')
        if abs((cpd['start'][selected][0]-start).to_value(u.s))>1e-3 or abs((cpd['end'][selected][-1]-end).to_value(u.s))>1e-3:
            raise ValueError('IDL boundaries do not match native edges')
        expected=reduced['counts'][selected,:,:,matches[0]].sum(axis=0)
        np.testing.assert_allclose(raw[k,:,:8],expected,rtol=1e-10,atol=1e-6)
        np.testing.assert_allclose(live_saved[k],live_s[selected].sum(axis=0),rtol=2e-6,atol=1e-5)
        r0=raw[k,detectors,:8].sum(axis=1)/live_saved[k,detectors]
        r1=corrected[k,detectors,:8].sum(axis=1)/live_saved[k,detectors]
        rows.append(dict(band_keV=f'{low:g}–{high:g}',native_start_utc=cpd['start'][selected][0].isot,
            native_end_utc=cpd['end'][selected][-1].isot,native_bins=int(selected.sum()),
            observed_counts=float(raw[k,detectors,:8].sum()),
            ELUT_corrected_counts=float(corrected[k,detectors,:8].sum()),
            raw_live_normalized_rate_s=float(r0.sum()),ELUT_live_normalized_rate_s=float(r1.sum()),
            summed_rate_change_percent=float(100*(r1.sum()/r0.sum()-1))))
        masks.append(selected)
    return dict(saved=saved,raw=raw,corrected=corrected,bands=bands,interval_masks=masks,
                summary=pd.DataFrame(rows))
