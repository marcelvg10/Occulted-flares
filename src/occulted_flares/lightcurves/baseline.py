"""Central detector-selection comparisons, without background or response inference.

No function accepts a final selection or writes a scientific output. Native
integration edges and reference-model live exposures remain the input contract.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def group_profiles(counts, duration_s, live_s, groups):
    """Sum pixels, correct each detector, then sum a named detector group.

    Counts: time × detector × pixel × band. Rates are summed counts/s,
    without area, energy-width, distance, background or energy-edge correction.
    No counting/livetime uncertainty independence is assumed here.
    """
    counts = np.asarray(counts, float)
    duration = np.asarray(duration_s, float)
    live = np.asarray(live_s, float)
    if (counts.ndim != 4 or duration.shape != (len(counts),)
            or live.shape != counts.shape[:2] or not np.isfinite(counts).all()
            or np.any(counts < 0) or not np.isfinite(duration).all() or np.any(duration <= 0)):
        raise ValueError('Expected finite nonnegative counts and positive native durations')
    detector_counts = counts.sum(axis=2)
    output = {}
    for name, selection in groups.items():
        indices = np.asarray(selection)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not len(indices) or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= counts.shape[1])):
            raise ValueError('Detector indices must be nonempty, unique, in-range integers')
        exposure = live[:, indices]
        if not np.isfinite(exposure).all() or np.any(exposure <= 0):
            raise ValueError('Invalid selected detector exposure: no bins silently omitted')
        c = detector_counts[:, indices, :]
        output[name] = dict(counts=c.sum(axis=1),
            observed_rate=c.sum(axis=1)/duration[:, None],
            live_rate=(c/exposure[..., None]).sum(axis=1), detector_indices=indices.copy())
    return output


def normalized_profiles(rate, selection):
    """Divide each band by its own maximum within the SAME chosen native bins.

    No peak interpolation or background subtraction. Return NaN for a band
    with zero maximum, rather than manufacturing a comparison profile.
    """
    rate = np.asarray(rate, float)
    selected = np.asarray(selection, bool)
    if (rate.ndim != 2 or selected.shape != (len(rate),) or not selected.any()
            or not np.isfinite(rate).all() or np.any(rate < 0)):
        raise ValueError('Finite nonnegative time × band rates and a nonempty selection required')
    peak = rate[selected].max(axis=0)
    return np.divide(rate, peak, out=np.full_like(rate, np.nan), where=peak > 0), peak


def positive_ratio(numerator, denominator):
    """Ratio where denominator is positive; zero-count denominators stay NaN."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matched finite arrays required')
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


def selection_summary(profiles, duration_s, selection, bands):
    """Observed counts and wall-time-weighted central rates over complete bins.

    The time-weighted corrected mean is sum_t[(sum_d C_dt/L_dt) T_t]/sum_t T_t.
    It differs from GSW's per-detector integrated live-weighted aggregation
    when flux and live fraction covary. Neither quantity is photon flux.
    """
    duration = np.asarray(duration_s, float)
    chosen = np.asarray(selection, bool)
    if (duration.ndim != 1 or chosen.shape != duration.shape or not chosen.any()
            or not np.isfinite(duration).all() or np.any(duration <= 0)):
        raise ValueError('Positive durations and a nonempty matched selection required')
    rows = []
    for name, p in profiles.items():
        if p['counts'].shape != (len(duration), len(bands)):
            raise ValueError('Profile shape differs from durations/bands')
        for k, (lo, hi) in enumerate(bands):
            rows.append(dict(group=name, detectors=len(p['detector_indices']),
                band_keV=f'{lo}–{hi}', raw_counts=float(p['counts'][chosen, k].sum()),
                observed_mean_counts_s=float(p['counts'][chosen, k].sum()/duration[chosen].sum()),
                time_weighted_live_rate_counts_s=float(np.average(
                    p['live_rate'][chosen, k], weights=duration[chosen]))))
    return pd.DataFrame(rows)


def check_audit_sources(work, event, audit):
    """Verify recorded source hashes when present; missing SSW is explicit.

    A changed source requires a fresh audit rather than repeating stale
    conclusions. Another checkout can still read the recorded audit without
    installing SolarSoft. Actual historical session state remains unknown.
    """
    rows = []
    for source in audit['sources']:
        root = Path(work) if source['root'] == 'work' else Path(event)
        p = root/source['path']
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != source['sha256']:
                raise ValueError(f'IDL source changed; re-audit before using conclusions: {p}')
            status = 'Current file hash matches recorded audit'
        else:
            status = 'File absent locally; recorded audit only'
        rows.append(dict(source=source['path'], status=status, sha256=source['sha256']))
    return pd.DataFrame(rows)
