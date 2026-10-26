"""Exploratory feature checks and draft interval accounting, without fitting."""
import numpy as np
from .background import complete_interval
from .excess import prepare_candidates, require_disjoint, combine_native, estimate_excess


def balanced_halves(reference, indices):
    """Alternate ordered resolution labels, retaining two disjoint TOP24 halves.

    A deterministic sensitivity split, not equal transmission or independence
    of common instrumental effects. No detector is accepted/rejected here.
    """
    ids = np.asarray(indices, int)
    if len(ids) < 2 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(reference['detectors'])):
        raise ValueError('At least two unique valid detector indices required')
    def key(i):
        label = reference['detectors'][i]['label']
        return int(label[:-1]), label[-1]
    ordered = np.array(sorted(ids, key=key), int)
    return dict(top24_half_A=ordered[::2], top24_half_B=ordered[1::2])


def prepare_view(context, indices):
    """Refit the SAME candidate windows for this detector selection."""
    return prepare_candidates(dict(context, indices=list(indices)))


def interval_measurement(product, prepared, start, end, background_names):
    """Whole native bins and background-conditional integrated measurements."""
    mask = complete_interval(product, start, end)
    ids = np.flatnonzero(mask)
    for name in background_names:
        require_disjoint(ids, prepared['masks'][name])
    data = combine_native(prepared['counts'], prepared['compression_variance'], prepared['live_s'], [ids])
    return dict(mask=mask, indices=ids, data=data,
        results={name: estimate_excess(**data, model=prepared['models'][name]) for name in background_names},
        actual_start=product['start'][ids[0]], actual_end=product['end'][ids[-1]],
        native_seconds=float(product['duration_s'][ids].sum()))


def interval_contrast(first, second, model):
    """Difference of mean rates in DISJOINT intervals, same constant background.

    Candidate background mean and its shared uncertainty cancel. The returned
    variance includes independent count/compression terms at fixed exposure
    only. It is not a feature-detection probability, and does not test drift.
    """
    if np.intersect1d(first['indices'], second['indices']).size:
        raise ValueError('Contrast intervals overlap; source covariance required')
    one = estimate_excess(**first['data'], model=model)
    two = estimate_excess(**second['data'], model=model)
    c1, l1 = first['data']['counts'][0], first['data']['live_s'][0, :, None]
    c2, l2 = second['data']['counts'][0], second['data']['live_s'][0, :, None]
    per_detector = c1/l1-c2/l2
    return dict(difference_rate=(one['excess_rate']-two['excess_rate'])[0],
        conditional_count_sigma=np.sqrt(one['count_variance']+two['count_variance'])[0],
        per_detector_difference=per_detector)


def signed_normalization(rate, selection):
    """Display scaling by own positive in-window maximum, preserving negatives."""
    values = np.asarray(rate, float)
    selected = np.asarray(selection, bool)
    if values.ndim != 2 or selected.shape != (len(values),) or not selected.any() or not np.isfinite(values).all():
        raise ValueError('Finite rates and a nonempty matched display window required')
    peak = values[selected].max(axis=0)
    return np.divide(values, peak, out=np.full_like(values, np.nan), where=peak > 0)


def interval_count_budget(measurement, models, background_names):
    """Raw and expected-background counts for an interval, per measured band.

    Range across candidate models is model sensitivity, not a confidence
    interval. Total counts alone cannot establish visibility/imaging quality.
    """
    counts, live = measurement['data']['counts'][0], measurement['data']['live_s'][0]
    backgrounds = np.stack([(models[name]['rate']*live[:, None]).sum(axis=0) for name in background_names])
    total = counts.sum(axis=0)
    fraction = np.divide(counts.max(axis=0), total, out=np.full_like(total, np.nan), where=total > 0)
    return dict(raw_counts=total, minimum_detector_counts=counts.min(axis=0),
        median_detector_counts=np.median(counts, axis=0), largest_detector_raw_count_fraction=fraction,
        background_counts_min=backgrounds.min(axis=0), background_counts_max=backgrounds.max(axis=0))
