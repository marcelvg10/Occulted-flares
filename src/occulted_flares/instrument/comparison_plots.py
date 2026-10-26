"""Visualize saved IDL results against Python; never change the calculations."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _paired_plot(time, expected, saved, fields, labels, detector_id,
                 selection, residual_scales, residual_labels, title):
    if not isinstance(detector_id, (int, np.integer)) or not 1 <= detector_id <= 32:
        raise ValueError('Use a one-based detector ID between 1 and 32')
    chosen = np.asarray(selection, bool)
    if chosen.shape != (len(time),) or not chosen.any():
        raise ValueError('Non-empty native-bin selection required')
    d = detector_id-1
    x = time[chosen].plot_date
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True,
                             layout='constrained', height_ratios=[2, 1])
    for column, (field, label, scale, residual_label) in enumerate(
            zip(fields, labels, residual_scales, residual_labels)):
        py, idl = np.asarray(expected[field]), np.asarray(saved[field])
        if py.shape != (len(time), 32) or idl.shape != py.shape:
            raise ValueError(f'Expected matching time × 32 arrays for {field}')
        a, b = py[chosen, d], idl[chosen, d]
        axes[0, column].plot(x, a, color='C0', lw=1.5, label='Python')
        axes[0, column].plot(x, b, color='C1', lw=1, ls='--', label='Saved IDL')
        axes[0, column].set(ylabel=label)
        axes[0, column].legend()
        axes[1, column].plot(x, (b-a)*scale, color='C3', lw=.9)
        axes[1, column].axhline(0, color='grey', lw=.8)
        axes[1, column].set(ylabel=residual_label, xlabel='Native arrival UTC, 7 July 2026')
        axes[1, column].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        axes[1, column].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    fig.suptitle(f'{title} — detector {detector_id}')
    return fig


def plot_central_comparison(time, fraction, live_s, saved, detector_id, selection):
    """Overlays plus explicitly scaled signed numerical residuals."""
    return _paired_plot(time, dict(lt_fraction=fraction, live_seconds=live_s),
        saved, ['lt_fraction', 'live_seconds'],
        ['Effective fraction f', 'Live exposure L [s]'], detector_id, selection,
        [1e9, 1e9], ['IDL − Python [×10⁻⁹, fraction]', 'IDL − Python [ns of exposure]'],
        'Central values: overlays and numerical residuals')


def plot_uncertainty_comparison(time, fraction_sigma, live_sigma_s, saved,
                                detector_id, selection):
    """Display discrepant uncertainty estimates without treating them as data noise."""
    return _paired_plot(time, dict(lt_sigma=fraction_sigma, live_sigma=live_sigma_s),
        saved, ['lt_sigma', 'live_sigma'],
        ['Estimated σ(f), dimensionless', 'Estimated σ(L) [s]'], detector_id, selection,
        [1, 1], ['IDL − Python, σ(f)', 'IDL − Python, σ(L) [s]'],
        'Uncertainty estimates: disagreement remains unresolved')


def plot_comparison_maps(starts, ends, fraction, fraction_sigma, saved):
    """All native bins/detectors: signed residual maps with real time-bin widths."""
    import astropy.units as u
    if len(starts) != len(ends) or not len(starts):
        raise ValueError('Non-empty matching time edges required')
    if np.any(np.abs((starts[1:]-ends[:-1]).to_value(u.s)) > 1e-6):
        raise ValueError('Plot requires contiguous integrations; split runs at real gaps')
    if np.any((ends-starts).to_value(u.s) <= 0):
        raise ValueError('Positive integration widths required')
    x = np.r_[starts[0].plot_date, ends.plot_date]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    for ax, field, py, title in zip(axes, ['lt_fraction', 'lt_sigma'],
            [fraction, fraction_sigma],
            ['Central fraction: numerical residual', 'Fraction uncertainty: unresolved discrepancy']):
        py, actual = np.asarray(py), np.asarray(saved[field])
        if py.shape != (len(starts), 32) or actual.shape != py.shape:
            raise ValueError(f'Expected matching time × 32 arrays for {field}')
        delta = actual-py
        finite = np.isfinite(delta)
        limit = np.max(np.abs(delta[finite])) if finite.any() else 0.
        limit = max(float(limit), 1e-18)
        im = ax.pcolormesh(x, np.arange(.5, 33.5), np.ma.masked_invalid(delta.T),
                          cmap='RdBu_r', vmin=-limit, vmax=limit,
                          shading='flat', rasterized=True)
        ax.set(ylabel='Detector ID (1-based)', yticks=[1, 4, 8, 12, 16, 20, 24, 28, 32], title=title)
        colorbar = fig.colorbar(im, ax=ax, label='IDL − Python, dimensionless')
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
    axes[-1].set_xlabel('Native Solar Orbiter-arrival UTC, 7 July 2026')
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.suptitle('Saved IDL versus Python — all 32 detectors; separate colour scales')
    return fig
