"""Finite-window comparison and timestamp consistency, independent of flux."""
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u


def merge_intervals(starts, ends):
    """Union positive half-open intervals, avoiding double-counted coverage."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if starts.shape != ends.shape or not np.isfinite([starts, ends]).all():
        raise ValueError("Finite, equally shaped edges required")
    result = []
    for a, b in sorted(zip(starts.ravel(), ends.ravel())):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return np.asarray(result, float).reshape(-1, 2)


def covered_seconds(start, end, union):
    if end <= start:
        return 0.
    return float(np.maximum(0., np.minimum(end, union[:, 1])-np.maximum(start, union[:, 0])).sum())


def coverage_envelopes(native_starts, native_ends, earliest_starts, latest_starts,
                       earliest_ends, latest_ends):
    """Keep shared edges together before constructing scenario coverage bounds.

    Each scenario uses one smooth, monotonic emission mapping for every STIX
    bin. Exactly touching native bins therefore form one continuous run in
    every scenario. Shrinking each short bin separately would invent holes.
    Only numerical equality (1 ns), not the audit's 1 us threshold, joins runs.
    Real gaps, overlaps or ordering anomalies conservatively start a new run.
    """
    n = len(native_starts)
    if not n:
        return merge_intervals([], []), merge_intervals([], [])
    breaks = np.flatnonzero(np.abs(np.asarray(native_starts)[1:]-np.asarray(native_ends)[:-1]) > 1e-9)+1
    first = np.r_[0, breaks]
    last = np.r_[breaks-1, n-1]
    inner = merge_intervals(np.asarray(latest_starts)[first], np.asarray(earliest_ends)[last])
    outer = merge_intervals(np.asarray(earliest_starts)[first], np.asarray(latest_ends)[last])
    return inner, outer


def table_edges(table, ephem, kind="emission", bound=None):
    """Use split TDB dates for nominal arithmetic; envelope strings retain us."""
    if kind == "emission" and bound is None:
        return tuple(ephem.seconds(Time(table[f"emission_{edge}_tdb_jd1"].to_numpy(),
            table[f"emission_{edge}_tdb_jd2"].to_numpy(), format="jd", scale="tdb")) for edge in ("start", "end"))
    suffix = "_"+bound if bound else ""
    return tuple(ephem.seconds(Time(table[f"{kind}_{edge}{suffix}_utc"].tolist(), scale="utc")) for edge in ("start", "end"))


def compare_exposures(tables, ephem):
    """Nominal coverage plus conservative bounds from independent edge extrema.

    Inner windows belong to every configured scenario; outer windows contain
    all of them. Bounds allow independent EUV/STIX hypotheses, not probability.
    Unknown clock offsets are not enclosed by these source-model bounds.
    """
    stix, euv = tables["stix"], tables["euvi"]
    s, e = table_edges(stix, ephem)
    a, b = table_edges(euv, ephem)
    slo, elo = table_edges(stix, ephem, bound="earliest")
    shi, ehi = table_edges(stix, ephem, bound="latest")
    alo, blo = table_edges(euv, ephem, bound="earliest")
    ahi, bhi = table_edges(euv, ephem, bound="latest")
    nominal = merge_intervals(s, e)
    ns, ne = table_edges(stix, ephem, "native")
    inner, outer = coverage_envelopes(ns, ne, slo, shi, elo, ehi)
    na, nb = table_edges(euv, ephem, "native")
    native = merge_intervals(ns, ne)
    rows, pairs = [], []
    for j, row in euv.iterrows():
        overlaps = np.maximum(0., np.minimum(b[j], e)-np.maximum(a[j], s))
        indices = np.flatnonzero(overlaps > 1e-9)
        lower = covered_seconds(ahi[j], blo[j], inner)
        upper = covered_seconds(alo[j], bhi[j], outer)
        coverage = covered_seconds(a[j], b[j], nominal)
        rows.append(dict(file=row.file, wavelength_A=row.wavelength_A,
            native_start_utc=row.native_start_utc, emission_start_utc=row.emission_start_utc,
            emission_end_utc=row.emission_end_utc, exposure_s=row.exposure_s,
            emission_window_s=b[j]-a[j], overlap_s=coverage,
            covered_fraction=coverage/(b[j]-a[j]), overlapping_stix_bins=len(indices),
            overlap_lower_bound_s=lower, overlap_upper_bound_s=upper,
            coverage_class="overlap in every configured scenario" if lower > 1e-9 else
                "no overlap in configured scenarios" if upper <= 1e-9 else "overlap not certified by conservative bounds",
            uncorrected_arrival_overlap_s=covered_seconds(na[j], nb[j], native)))
        for i in indices:
            pairs.append(dict(euvi_file=row.file, native_stix_bin=int(stix.iloc[i].native_bin),
                overlap_s=overlaps[i], stix_bin_span_s=e[i]-s[i],
                stix_bin_fraction=overlaps[i]/(e[i]-s[i])))
    return pd.DataFrame(rows), pd.DataFrame(pairs, columns=["euvi_file", "native_stix_bin", "overlap_s", "stix_bin_span_s", "stix_bin_fraction"])


def audit_timestamps(tables):
    """Internal checks only: never infer clock accuracy from regular sampling."""
    native = tables["stix_native"]
    starts, ends = native["start"], native["end"]
    durations = native["duration_s"]
    gaps = (starts[1:]-ends[:-1]).to_value(u.s)
    centres = starts+(ends-starts)/2
    step = (centres[1:]-centres[:-1]).to_value(u.s)
    tol = 1e-6
    transitions = pd.DataFrame(dict(after_native_bin=np.arange(len(gaps)),
        next_start_minus_previous_end_s=gaps, centre_step_s=step))
    h = native["provenance"]["header"]
    summary = pd.DataFrame([
        dict(check="Native STIX bins", value=len(starts)),
        dict(check="Nonpositive integration durations", value=int(np.sum(durations <= 0))),
        dict(check="Non-increasing bin centres", value=int(np.sum(step <= 0))),
        dict(check="Positive gaps (>1 microsecond)", value=int(np.sum(gaps > tol))),
        dict(check="Overlapping neighbours (>1 microsecond)", value=int(np.sum(gaps < -tol))),
        dict(check="Minimum / maximum duration [s]", value=f"{durations.min():g} / {durations.max():g}"),
        dict(check="First edge minus DATE-BEG [s]", value=float((starts[0]-Time(h["DATE-BEG"], scale=native["provenance"]["time_scale"])).to_value(u.s))),
        dict(check="Last edge minus DATE-END [s]", value=float((ends[-1]-Time(h["DATE-END"], scale=native["provenance"]["time_scale"])).to_value(u.s))),
    ])
    return summary, transitions, pd.DataFrame({"native_duration_s":np.round(durations, 6)}).value_counts().rename("bins").reset_index()


def timing_header_inventory(path):
    """Report present/absent timing and processing metadata, without defaults."""
    from astropy.io import fits
    keys = ("TIMESYS", "TIMEREF", "TREFPOS", "TASSIGN", "TIMSYER", "TIMRDER", "CLOCKAPP", "DATE-BEG", "DATE-END", "EAR_TDEL", "SUN_TIME", "VERS_SW", "VERSION", "CREATOR")
    with fits.open(path, memmap=True) as hdus:
        return pd.DataFrame([dict(hdu=hdu.name, keyword=k, present=k in hdu.header,
            value=str(hdu.header[k]) if k in hdu.header else "not supplied")
            for hdu in hdus[:3] for k in keys])


def plot_comparison(comparison, tables, ephem, focus_file):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(len(comparison))
    axes[0].barh(y, comparison.covered_fraction, color=["tab:red" if w==304 else "tab:green" for w in comparison.wavelength_A])
    axes[0].set_yticks(y, [f"{int(r.wavelength_A)} A | {r.native_start_utc[11:19]}" for r in comparison.itertuples()], fontsize=8)
    axes[0].set_xlim(0,1.08)
    axes[0].set_xlabel("Fraction of EUVI emission window covered by STIX")
    axes[0].set_title("All downloaded exposures; nominal source models")
    stix, euv = tables["stix"], tables["euvi"]
    s,e = table_edges(stix,ephem)
    a,b = table_edges(euv,ephem)
    j = int(np.flatnonzero(euv.file.to_numpy()==focus_file)[0])
    zero = (a[j]+b[j])/2
    indices = np.flatnonzero((e>zero-6)&(s<zero+6))
    axes[1].broken_barh([(s[i]-zero,e[i]-s[i]) for i in indices],(-.2,.4), facecolors="tab:blue",edgecolors="white",linewidth=.5)
    axes[1].broken_barh([(a[j]-zero,b[j]-a[j])],(.8,.4),facecolors="tab:red")
    alo,blo=table_edges(euv,ephem,bound="earliest")
    ahi,bhi=table_edges(euv,ephem,bound="latest")
    axes[1].broken_barh([(alo[j]-zero,bhi[j]-alo[j])],(1.8,.4),facecolors="gray",alpha=.35)
    axes[1].set_yticks([0,1,2],["STIX native bins\nconverted to emission time","EUVI nominal exposure","EUVI outer scenario window"])
    axes[1].set_xlim(-6,6)
    axes[1].set_xlabel("Seconds from nominal EUVI emission midpoint")
    axes[1].set_title(f"{focus_file}\nNo resampling or fractional-count allocation",fontsize=10)
    for ax in axes:
        ax.grid(axis="x",alpha=.25)
    fig.tight_layout()
    return fig


def plot_timestamp_audit(tables, transitions):
    import matplotlib.pyplot as plt
    native=tables["stix_native"]
    minutes=(native["start"]-native["start"][0]).to_value(u.s)/60
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].step(minutes,native["duration_s"],where="post")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Native integration [s]")
    axes[0].set_title("Recorded bin durations: variable sampling, not uniform 0.3 s coverage")
    axes[1].plot(minutes[1:],transitions.next_start_minus_previous_end_s*1000)
    extent = max(1., float(np.max(np.abs(transitions.next_start_minus_previous_end_s)))*1100)
    axes[1].set_ylim(-extent, extent)
    axes[1].set_title("Continuity diagnostic; sub-microsecond arithmetic residuals are not clock measurements", fontsize=10)
    axes[1].set_ylabel("Next start - prior end [ms]")
    axes[1].set_xlabel("Minutes since first native STIX edge")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout()
    return fig
