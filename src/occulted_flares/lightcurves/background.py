"""Candidate background diagnostics conditional on fixed native live exposures.

These are exploration tools, not a complete signal detection likelihood.
No candidate is auto-accepted. Source/trigger covariance, livetime uncertainty,
energy calibration and possible temporal background drift remain external.
"""
import numpy as np


def complete_interval(product, start, end):
    """Select only fully contained integrations; do not apportion edge counts."""
    from astropy.time import Time
    a, b = Time(start), Time(end)
    if not a < b:
        raise ValueError('Interval end must follow start')
    chosen = (product['start'] >= a) & (product['end'] <= b)
    if not chosen.any():
        raise ValueError('No complete native bins in candidate interval')
    return chosen


def fit_constant_background(counts, compression_variance, live_s, selection):
    """Per-detector/band Poisson MLE sum C/sum L, with compression diagnostic.

    Inputs: time × detector × band, and time × detector live seconds. Poisson
    MLE is conditional on known exposure and constant background in selection.
    Compression is additional approximate variance; it is not a Poisson count.
    Variance estimate sum(C+compression_variance)/(sum L)^2 is a plug-in
    diagnostic only. Zero-count cells get zero plug-in variance and are
    explicitly flagged; they require proper interval/likelihood treatment.
    """
    c = np.asarray(counts, float)
    comp = np.asarray(compression_variance, float)
    live = np.asarray(live_s, float)
    chosen = np.asarray(selection, bool)
    if (c.ndim != 3 or comp.shape != c.shape or live.shape != c.shape[:2]
            or chosen.shape != (len(c),) or not chosen.any()
            or not np.isfinite(c).all() or np.any(c < 0)
            or not np.isfinite(comp).all() or np.any(comp < 0)
            or not np.isfinite(live).all() or np.any(live <= 0)):
        raise ValueError('Matched finite nonnegative count/variance arrays and positive exposure required')
    total = c[chosen].sum(axis=0)
    exposure = live[chosen].sum(axis=0)[:, None]
    variance = (total+comp[chosen].sum(axis=0))/exposure**2
    return dict(rate=total/exposure, variance=variance, counts=total,
                live_s=exposure[:, 0], zero_count_cells=total == 0,
                native_bins=int(chosen.sum()))


def predict_background(model, live_s):
    """Expected native detector counts and summed rate for a candidate model.

    The model rate is constant. Its uncertainty is shared across all predicted
    times, not an independent error to reduce by summing more science bins.
    """
    live = np.asarray(live_s, float)
    rate, var = np.asarray(model['rate']), np.asarray(model['variance'])
    if (live.ndim != 2 or live.shape[1] != len(rate) or rate.shape != var.shape
            or not np.isfinite(live).all() or np.any(live <= 0)):
        raise ValueError('Prediction exposure does not match fitted detectors')
    return dict(expected_counts=(live[..., None]*rate[None]).sum(axis=1),
                summed_rate=rate.sum(axis=0),
                summed_rate_variance=var.sum(axis=0))


def count_residual_diagnostic(counts, compression_variance, live_s, model):
    """(Observed-expected)/sqrt(expected+compression variance) in raw count space.

    Omits fitted-background uncertainty and uses the same data if evaluated
    in fitting bins. No normal p-value, degrees of freedom or formal goodness
    of fit is implied. Zero expected variance remains NaN.
    """
    expected = predict_background(model, live_s)['expected_counts']
    c = np.asarray(counts, float).sum(axis=1)
    comp = np.asarray(compression_variance, float).sum(axis=1)
    denom = np.sqrt(expected+comp)
    return np.divide(c-expected, denom, out=np.full_like(expected, np.nan), where=denom > 0)


def overlap_aggregate(product, counts, live_s, starts, ends):
    """Reproduce integrated native counts weighted by mean live exposure.

    For exact complete-bin groups only. First form native summed detector rate
    R_t, then pseudo-counts R_t * mean_d(L_dt). Integrated OSPEX rate is their
    sum divided by summed mean live exposure. Reject partial edge matching.
    This is distinct from wall-time or per-detector integrated aggregation.
    """
    from astropy.time import Time
    c = np.asarray(counts, float)
    l = np.asarray(live_s, float)
    if c.ndim != 3 or l.shape != c.shape[:2] or np.any(l <= 0) or not np.isfinite(l).all():
        raise ValueError('Matched time/detector/band counts and positive exposures required')
    native_rate = (c/l[..., None]).sum(axis=1)
    average_live = l.mean(axis=1)
    output, members = [], []
    for a, b in zip(Time(starts), Time(ends)):
        # IDL anytim may serialize bin edges to milliseconds.
        inside = ((product['start']-a).to_value('s') >= -0.0011) & ((product['end']-b).to_value('s') <= 0.0011)
        indices = np.flatnonzero(inside)
        if (not len(indices) or abs((product['start'][indices[0]]-a).to_value('s')) > 0.0011
                or abs((product['end'][indices[-1]]-b).to_value('s')) > 0.0011
                or np.any(np.diff(indices) != 1)):
            raise ValueError('IDL interval does not match complete consecutive native bins')
        if len(indices) > 1 and np.any(np.abs((product['start'][indices[1:]]-product['end'][indices[:-1]]).to_value('s')) > 1e-6):
            raise ValueError('A matched interval crosses a native gap')
        output.append(np.average(native_rate[inside], axis=0, weights=average_live[inside]))
        members.append(indices)
    return np.asarray(output), members
