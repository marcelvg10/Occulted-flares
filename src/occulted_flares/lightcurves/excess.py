"""Conditional excess estimates with background alternatives kept separate.

Known live exposure, count/compression independence and a constant candidate
background are assumptions. No formal detection probability or confidence
interval follows from the diagnostic variance returned here.
"""
import numpy as np
from .background import complete_interval, fit_constant_background


def prepare_candidates(context):
    indices = context['indices']
    counts = context['reduced']['counts'][:, indices].sum(axis=2)
    comp = context['reduced']['compression_variance'][:, indices].sum(axis=2)
    live = context['live'][:, indices]
    masks, models = {}, {}
    for candidate in context['settings']['background_candidates']:
        name = candidate['name']
        if name in masks:
            raise ValueError('Duplicate background candidate name')
        masks[name] = complete_interval(context['cpd'], candidate['start'], candidate['end'])
        models[name] = fit_constant_background(counts, comp, live, masks[name])
    return dict(counts=counts, compression_variance=comp, live_s=live,
                masks=masks, models=models)


def estimate_excess(counts, compression_variance, live_s, model):
    """Native or aggregated time × detector × band measurements.

    Sum_d(C_d/L_d - b_d), variance sum_d[(C_d+comp_d)/L_d² + var(b_d)].
    The source interval MUST be disjoint from fitted background data when
    this independent-error expression is used. Callers verify native indices.
    Background variance is shared across time, never divided by bin count.
    """
    c, comp, live = [np.asarray(v, float) for v in (counts, compression_variance, live_s)]
    rate, var = np.asarray(model['rate'], float), np.asarray(model['variance'], float)
    if (c.ndim != 3 or comp.shape != c.shape or live.shape != c.shape[:2]
            or rate.shape != c.shape[1:] or var.shape != rate.shape
            or not all(np.isfinite(v).all() for v in (c, comp, live, rate, var))
            or any(np.any(v < 0) for v in (c, comp, rate, var)) or np.any(live <= 0)):
        raise ValueError('Matched nonnegative counts/variance/model and positive exposures required')
    observed = (c/live[..., None]).sum(axis=1)
    count_variance = ((c+comp)/live[..., None]**2).sum(axis=1)
    background = rate.sum(axis=0)
    background_variance = var.sum(axis=0)
    excess = observed-background
    sigma = np.sqrt(count_variance+background_variance)
    relative = np.divide(sigma, excess, out=np.full_like(excess, np.nan), where=excess > 0)
    return dict(observed_rate=observed, background_rate=background, excess_rate=excess,
                count_variance=count_variance, background_variance=background_variance,
                conditional_sigma=sigma, positive_excess=excess > 0,
                conditional_relative_error=relative)


def require_disjoint(source_indices, background_mask):
    indices = np.asarray(source_indices)
    mask = np.asarray(background_mask, bool)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(mask)) or not len(indices)):
        raise ValueError('Nonempty valid native indices required')
    if mask[indices].any():
        raise ValueError('Source and fitted background overlap: covariance cannot be ignored')


def combine_native(counts, compression_variance, live_s, groups):
    """Conserve counts and live seconds. No fractional-bin assignment."""
    c, comp, live = [np.asarray(v, float) for v in (counts, compression_variance, live_s)]
    if (c.ndim != 3 or comp.shape != c.shape or live.shape != c.shape[:2]
            or not all(np.isfinite(v).all() for v in (c, comp, live))
            or np.any(c < 0) or np.any(comp < 0) or np.any(live <= 0)):
        raise ValueError('Mismatched count/exposure arrays')
    flattened = []
    output_c, output_v, output_l = [], [], []
    for group in groups:
        g = np.asarray(group)
        if (g.ndim != 1 or not len(g) or not np.issubdtype(g.dtype, np.integer)
                or np.any(g < 0) or np.any(g >= len(c)) or np.any(np.diff(g) != 1)):
            raise ValueError('Each group must contain consecutive valid native indices')
        flattened.extend(g.tolist())
        output_c.append(c[g].sum(axis=0))
        output_v.append(comp[g].sum(axis=0))
        output_l.append(live[g].sum(axis=0))
    if not flattened or len(set(flattened)) != len(flattened) or np.any(np.diff(flattened) <= 0):
        raise ValueError('Nonempty nonoverlapping chronological groups required')
    return dict(counts=np.asarray(output_c), compression_variance=np.asarray(output_v),
                live_s=np.asarray(output_l))
