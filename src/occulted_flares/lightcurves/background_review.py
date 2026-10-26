"""Window-by-window display diagnostics using unchanged candidate models."""
import numpy as np
import pandas as pd

from .background import count_residual_diagnostic


def window_table(product, counts, compression_variance, live_s, selection, model, bands):
    """Return native bins, rates and conditional residuals without refitting.

    Rate error bars include counts plus compression variance at fixed live
    exposure. They omit livetime/model errors and fitted-background uncertainty.
    Residuals retain background.py's count-space definition, including its
    fitting-data dependence. No significance, fit p-value or rejection follows.
    """
    c, comp, live = map(lambda x: np.asarray(x, float), (counts, compression_variance, live_s))
    chosen = np.asarray(selection, bool)
    if (c.ndim != 3 or c.shape != comp.shape or live.shape != c.shape[:2]
            or chosen.shape != (len(c),) or not chosen.any() or len(bands) != c.shape[-1]
            or len(product['time']) != len(c) or not np.isfinite(c).all() or np.any(c < 0)
            or not np.isfinite(comp).all() or np.any(comp < 0)
            or not np.isfinite(live).all() or np.any(live <= 0)):
        raise ValueError('Matched counts, variances, positive exposures and nonempty selection required')
    residual = count_residual_diagnostic(c, comp, live, model)
    rate = (c/live[..., None]).sum(axis=1)
    sigma = np.sqrt(((c+comp)/live[..., None]**2).sum(axis=1))
    rows = []
    origin = product['start'][chosen][0]
    for i in np.flatnonzero(chosen):
        for k, (lo, hi) in enumerate(bands):
            rows.append(dict(native_bin=int(i), band_keV=f'{lo}–{hi}', band_index=k,
                start_native_UTC=product['start'][i].isot, end_native_UTC=product['end'][i].isot,
                start_offset_s=float((product['start'][i]-origin).to_value('s')),
                end_offset_s=float((product['end'][i]-origin).to_value('s')),
                raw_counts=float(c[i, :, k].sum()), rate_counts_s=rate[i, k],
                conditional_rate_sigma_counts_s=sigma[i, k],
                fitted_background_counts_s=float(model['rate'][:, k].sum()),
                count_residual=residual[i, k]))
    return pd.DataFrame(rows)


def plot_window(table, bands, name, residual_limit):
    """Rates and residuals on each window's own time axis; no data exports."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(bands), figsize=(16, 7), sharex='col',
                             squeeze=False, layout='constrained')
    for k, (lo, hi) in enumerate(bands):
        rows = table[table.band_index == k]
        x0, x1 = rows.start_offset_s.to_numpy(), rows.end_offset_s.to_numpy()
        x, dx = (x0+x1)/2, (x1-x0)/2
        top, bottom = axes[:, k]
        top.hlines(rows.rate_counts_s, x0, x1, color='C0', lw=1.5)
        top.errorbar(x, rows.rate_counts_s, yerr=rows.conditional_rate_sigma_counts_s,
                     fmt='o', color='C0', ms=4, capsize=3, label='Native rate + conditional error')
        top.axhline(rows.fitted_background_counts_s.iloc[0], color='C3', ls='--', label='Fitted constant')
        top.set(title=f'{lo}–{hi} keV', ylabel='Summed counts / s')
        bottom.errorbar(x, rows.count_residual, xerr=dx, fmt='o', ms=4, capsize=2, color='C0')
        bottom.axhline(0, color='0.5', lw=.8)
        bottom.set(ylabel='Conditional count residual', xlabel='Seconds from window start',
                   ylim=(-residual_limit, residual_limit))
        for ax in (top, bottom):
            ax.grid(alpha=.2)
            ax.set_xlim(-1, x1.max()+1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{name}: {table.start_native_UTC.iloc[0]} to {table.end_native_UTC.iloc[-1]} UTC\n'
                 'Native bin widths preserved; shared residual scale across windows; no rejection threshold')
    return fig
