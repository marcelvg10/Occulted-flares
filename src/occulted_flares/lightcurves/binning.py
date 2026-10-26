"""Whole-native-bin grouping for display and conditional precision exploration.

Adaptive stopping depends on the measured counts. Its bins are not independent
pre-specified tests and do not establish a physical variability timescale.
"""
import numpy as np
import pandas as pd
from .excess import estimate_excess


def _selection(product, selection):
    chosen = np.asarray(selection, bool)
    if chosen.shape != (len(product['start']),) or not chosen.any():
        raise ValueError('Nonempty matched native selection required')
    ids = np.flatnonzero(chosen)
    duration = np.asarray(product['duration_s'], float)
    if (duration.shape != chosen.shape or not np.isfinite(duration[ids]).all()
            or np.any(duration[ids] <= 0) or np.any(product['end'][ids] <= product['start'][ids])):
        raise ValueError('Selected native durations and edges must be positive')
    if np.any(np.diff(ids) != 1):
        raise ValueError('Use one consecutive selected run; never bridge unselected bins')
    if len(ids) > 1 and np.any(np.abs((product['start'][ids[1:]]-product['end'][ids[:-1]]).to_value('s')) > 1e-6):
        raise ValueError('Selection crosses a gap or overlap')
    return ids


def fixed_groups(product, selection, target_s, phase_native_bins=0):
    """Minimum target duration with 1 ns arithmetic tolerance, whole bins only.

    A phase shift leaves an explicit leading partial group; the final short
    group is also retained. The native acquisition never becomes finer.
    """
    ids = _selection(product, selection)
    if (not np.isfinite(target_s) or target_s <= 0 or not isinstance(phase_native_bins, (int, np.integer))
            or phase_native_bins < 0 or phase_native_bins >= len(ids)):
        raise ValueError('Positive target and valid integer phase required')
    groups = []
    if phase_native_bins:
        groups.append(ids[:phase_native_bins])
    buffer, total = [], 0.
    for i in ids[phase_native_bins:]:
        buffer.append(i)
        total += product['duration_s'][i]
        if total >= target_s-1e-9:
            groups.append(np.asarray(buffer, int))
            buffer, total = [], 0.
    if buffer:
        groups.append(np.asarray(buffer, int))
    return groups


def group_table(product, groups, target_s=None):
    rows = []
    for g in groups:
        width = float((product['end'][g[-1]]-product['start'][g[0]]).to_value('s'))
        rows.append(dict(start_native_UTC=product['start'][g[0]].isot,
            end_native_UTC=product['end'][g[-1]].isot, duration_s=width, native_bins=len(g),
            start_plot_date=product['start'][g[0]].plot_date,
            end_plot_date=product['end'][g[-1]].plot_date,
            shorter_than_target=bool(target_s is not None and width < target_s-1e-6)))
    return pd.DataFrame(rows)


def adaptive_groups(product, selection, counts, compression_variance, live_s, models,
                    band, relative_target=.2, maximum_s=9.):
    """Stop at positive excess and conditional sigma/excess <= target for ALL models.

    Cumulative counts, compression variance and live exposure are summed per
    detector. Maximum duration is a whole-bin cap (a long native bin cannot
    be split). Retain all bins and record target failures, including the tail.
    Callers verify source/background independence before using this criterion.
    """
    ids = _selection(product, selection)
    c, comp, live = [np.asarray(v, float) for v in (counts, compression_variance, live_s)]
    if (not models or not 0 < relative_target < 1 or not np.isfinite(maximum_s) or maximum_s <= 0
            or c.ndim != 3 or len(c) != len(product['start']) or comp.shape != c.shape or live.shape != c.shape[:2]
            or not 0 <= band < c.shape[2]):
        raise ValueError('Valid data, models, band, precision and duration cap required')
    if np.max(np.asarray(product['duration_s'])[ids]) > maximum_s+1e-9:
        raise ValueError('Maximum duration is below a native integration; cannot split it')
    groups, records, buffer = [], [], []
    cs, vs, ls, duration = None, None, None, 0.
    def assess():
        ratios = []
        for model in models.values():
            result = estimate_excess(cs[None], vs[None], ls[None], model)
            value = result['conditional_relative_error'][0, band]
            ratios.append(value if np.isfinite(value) else np.inf)
        worst = max(ratios)
        return worst <= relative_target, worst
    def close(reason, met, worst):
        groups.append(np.asarray(buffer, int))
        records.append(dict(reason=reason, target_met=bool(met),
                            worst_conditional_relative_error=worst))
    for pos, i in enumerate(ids):
        dt = product['duration_s'][i]
        if buffer and duration+dt > maximum_s+1e-9:
            met, worst = assess()
            close('duration_cap', met, worst)
            buffer, cs, vs, ls, duration = [], None, None, None, 0.
        buffer.append(i)
        cs = c[i].copy() if cs is None else cs+c[i]
        vs = comp[i].copy() if vs is None else vs+comp[i]
        ls = live[i].copy() if ls is None else ls+live[i]
        duration += dt
        met, worst = assess()
        if met:
            close('target_met', True, worst)
            buffer, cs, vs, ls, duration = [], None, None, None, 0.
        elif pos == len(ids)-1:
            close('end_of_run', False, worst)
            buffer = []
    return groups, pd.DataFrame(records)
