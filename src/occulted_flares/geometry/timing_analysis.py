"""Event orchestration for reproducible geometric photon-emission intervals."""
import json
from pathlib import Path
import numpy as np
import astropy.units as u
from dataclasses import replace
from occulted_flares.geometry.transform_coordinates import read_selection
from occulted_flares.geometry.timing_ephemeris import Ephemerides
from occulted_flares.geometry.photon_timing import infer_roots, read_stix_intervals, read_euvi_intervals, convert_intervals, solve_emission, solve_arrival


def setup(config_path, settings=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text()) if settings is None else settings
    base = config_path.parent
    _, selected, m, epoch, digest, messages = read_selection(base/config["selection_config"])
    ephemerides = Ephemerides(base/config["ephemeris_cache"])
    nominal_euv = infer_roots(m, ephemerides, *config["nominal_root_hpc_arcsec"], config["nominal_euv_feature_height_km"])
    nominal_stix = nominal_euv.with_heights(config["nominal_stix_source_height_km"])
    hypotheses = json.loads((base/config["assessment_config"]).read_text())
    locations = [dict(label="nominal", tx=config["nominal_root_hpc_arcsec"][0], ty=config["nominal_root_hpc_arcsec"][1])]
    locations += [dict(label=row["label"], tx=float(row["tx_arcsec"]), ty=float(row["ty_arcsec"])) for row in selected]
    for region in hypotheses["regions"]:
        x0, x1, y0, y1 = region["bounds_arcsec"]
        for x in np.linspace(x0, x1, config["region_grid_size"]):
            for y in np.linspace(y0, y1, config["region_grid_size"]):
                locations.append(dict(label=region["id"], tx=float(x), ty=float(y)))
    cases = []
    for location in locations:
        for dx, dy in config["pointing_shifts_arcsec"]:
            for height in config["euv_height_tests_km"]:
                cases.append(dict(location=location["label"], tx_arcsec=location["tx"]+dx,
                                  ty_arcsec=location["ty"]+dy, shift_x_arcsec=dx, shift_y_arcsec=dy, euv_height_km=height))
    euv_cases = infer_roots(m, ephemerides, [r["tx_arcsec"] for r in cases], [r["ty_arcsec"] for r in cases], [r["euv_height_km"] for r in cases])
    stix_cases = euv_cases.repeat_at_heights(config["stix_height_tests_km"])
    return dict(config=config, base=base, map=m, selected=selected, epoch=epoch, input_digest=digest,
                ephemerides=ephemerides, nominal_euv=nominal_euv, nominal_stix=nominal_stix,
                euv_cases=euv_cases, stix_cases=stix_cases, scenario_roots=cases, warnings=messages)


def interval_tables(state):
    print("Converting raw STIX bin edges and all configured source scenarios...", flush=True)
    config, base, ephem = state["config"], state["base"], state["ephemerides"]
    stix = read_stix_intervals(base/config["stix_input"])
    euv = read_euvi_intervals(base/config["euvi_directory"])
    stix_table = convert_intervals(stix, ephem, ephem.bodies["Solar Orbiter"], state["nominal_stix"], state["stix_cases"])
    stix_table.insert(0, "native_bin", np.arange(len(stix_table)))
    print("Converting EUVI exposure edges...", flush=True)
    euv_table = convert_intervals(euv, ephem, ephem.bodies["STEREO-A"], state["nominal_euv"], state["euv_cases"])
    for name in ("file", "wavelength_A", "exposure_s", "edge_span_s", "time_scale", "scale_source", "edge_source"):
        euv_table.insert(0, name, [r[name] for r in euv["rows"]])
    return dict(stix=stix_table, euvi=euv_table, stix_native=stix, euvi_native=euv)


def reference_event(state):
    """Predicted arrival times for one common emission; Earth is hypothetical."""
    import pandas as pd
    ephem = state["ephemerides"]
    source = state["nominal_stix"]
    emission = float(state["nominal_euv"].anchor_seconds[0])
    arrivals = {name: float(solve_arrival([emission], ephem.bodies[name], source)[0,0])
                for name in ("Solar Orbiter", "STEREO-A", "Earth")}
    centre = replace(source, radius_km=0., heights_km=np.array([0.]))
    centre_arrivals = {name: float(solve_arrival([emission], ephem.bodies[name], centre)[0,0]) for name in arrivals}
    rows = []
    for name, arrival in arrivals.items():
        recovered = solve_emission([arrival], ephem.bodies[name], source)[0,0]
        # Opaque-sphere clearance evaluated in the source-emission solar frame.
        p = source.position(np.array([[emission]]))[:,0,0]-ephem.bodies["Sun"].position(emission)
        o = ephem.bodies[name].position(arrival)-ephem.bodies["Sun"].position(emission)
        ray = o-p
        fraction = np.clip(-np.dot(p,ray)/np.dot(ray,ray), 0, 1)
        clearance = np.linalg.norm(p+fraction*ray)-source.radius_km
        rows.append(dict(observer=name, common_emission_utc=ephem.times(emission).utc.isot,
                         hypothetical_arrival_utc=ephem.times(arrival).utc.isot,
                         source_light_time_s=arrival-emission,
                         arrival_minus_solo_s=arrival-arrivals["Solar Orbiter"],
                         centre_arrival_minus_solo_s=centre_arrivals[name]-centre_arrivals["Solar Orbiter"],
                         source_minus_centre_delay_s=(arrival-emission)-(centre_arrivals[name]-emission),
                         roundtrip_residual_s=float(recovered-emission), geometric_ray_clear=bool(clearance >= -1e-5)))
    return pd.DataFrame(rows)


def plot_timing(tables, state):
    import matplotlib.pyplot as plt
    ephem = state["ephemerides"]
    stix = tables["stix"]
    offsets = (ephem.seconds(tables["stix_native"]["start"])-ephem.seconds(tables["stix_native"]["start"])[0])/60
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].plot(offsets, stix.start_light_time_s)
    axes[0].set_xlabel("Minutes since first native STIX bin")
    axes[0].set_ylabel("Source-to-Solar-Orbiter light time [s]")
    axes[0].set_title("Time-varying nominal correction")
    axes[0].ticklabel_format(useOffset=False, axis="y")
    axes[1].plot(offsets, stix.start_scenario_spread_s, label="Source-scenario envelope width")
    axes[1].axhline(0.3, color="gray", ls="--", label="0.3 s comparison scale")
    axes[1].set_xlabel("Minutes since first native STIX bin")
    axes[1].set_ylabel("Emission-time spread [s]")
    axes[1].set_title("Location/height/pointing sensitivity, not a confidence interval")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Straight-ray source-model timing; clock, calibration and relativistic uncertainties not fitted", fontsize=12)
    fig.tight_layout()
    return fig


def plot_exposure_windows(tables):
    """Show finite sampling windows; do not interpolate EUVI into a time series."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from astropy.time import Time
    from datetime import timezone
    fig, axes = plt.subplots(2, 1, figsize=(14,7), sharex=True)
    for ax, convention in zip(axes, ("native", "emission")):
        groups = [("STIX", tables["stix"], "tab:blue")]
        for wave, color in ((195,"tab:green"),(304,"tab:red")):
            groups.append((f"EUVI {wave}", tables["euvi"][tables["euvi"].wavelength_A == wave], color))
        for index, (name, table, color) in enumerate(groups):
            starts = dates.date2num(Time(table[convention+"_start_utc"].tolist(), scale="utc").to_datetime(timezone=timezone.utc))
            ends = dates.date2num(Time(table[convention+"_end_utc"].tolist(), scale="utc").to_datetime(timezone=timezone.utc))
            ax.broken_barh(list(zip(starts,ends-starts)), (index-.25,.5), facecolors=color)
        ax.set_yticks(range(3), [g[0] for g in groups])
        ax.xaxis.set_major_locator(dates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S", tz=timezone.utc))
        ax.tick_params(axis="x",labelbottom=True)
        ax.set_title("Native observer-arrival UTC" if convention == "native" else "Model photon-emission event, rendered UTC; nominal source assumptions")
        ax.grid(axis="x",alpha=.25)
    axes[-1].set_xlabel("UTC labels (not local time); bars are finite acquisition/exposure windows")
    fig.suptitle("Sampling windows, not measured light curves; coincidence/overlap tests belong to step 09", fontsize=12)
    fig.tight_layout()
    return fig
