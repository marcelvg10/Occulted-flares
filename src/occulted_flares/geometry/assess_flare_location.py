"""Guided location assessment and conditional occultation thresholds.

Direct VS Code Play works. Fixed outputs/location_assessment. Manual regions,
pointing shifts and EUV heights are explicit hypotheses in an event JSON; they
are NOT measured ribbons, calibrated errors, or a posterior distribution.
"""
import argparse
import json
from pathlib import Path

from occulted_flares.geometry.geometry_runtime import configure_cache, default_geometry_config, default_output
from occulted_flares.geometry.geometry_ephemeris import load_observers
from occulted_flares.geometry.coordinate_transforms import source_coordinate
from occulted_flares.geometry.occultation_geometry import radial_occultation
from occulted_flares.geometry.transform_coordinates import read_selection, write_table
from occulted_flares.geometry.euvi_context import context_montage


def plot_ranges(summaries, point_rows, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, observer in zip(axes, ("Solar Orbiter", "Earth")):
        selected = [r for r in summaries if r["observer"] == observer]
        for index, row in enumerate(selected):
            lo, hi = row["h_occ_min_km"]/1000, row["h_occ_max_km"]/1000
            ax.plot([lo, hi], [index, index], "o-", color="tab:blue", lw=3)
            ax.annotate(f"{lo:.2f}-{hi:.2f} Mm", ((lo+hi)/2, index), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        ax.set_yticks(range(len(selected)), [r["region"] for r in selected])
        ax.set_ylim(-0.5, max(1.5, len(selected)-0.5))
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Minimum visible radial height above photosphere [Mm]")
        ax.set_title(observer)
    fig.suptitle("Conditional occultation heights across location / pointing / EUV-height scenarios", fontsize=12)
    fig.text(0.5, 0.025, "Full scenario envelope, NOT a confidence interval or actual X-ray source height. Adopted opaque photosphere; source association provisional.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.18, wspace=0.4)
    ranges_figure = fig
    if output is not None:
        fig.savefig(output/"conditional_height_ranges.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, observer in zip(axes, ("Solar Orbiter", "Earth")):
        rows = [r for r in point_rows if r["observer"] == observer]
        ax.scatter([r["theta_occ_deg"] for r in rows], [r["h_occ_km"]/1000 for r in rows])
        for r in rows:
            ax.annotate(r["label"].replace("point_", ""), (r["theta_occ_deg"], r["h_occ_km"]/1000), xytext=(5, 6), textcoords="offset points", fontsize=9)
        ax.axvline(0, color="gray", ls=":")
        ax.set_xlabel("Geometric angle beyond 90-degree plane [deg]")
        ax.set_ylabel("Minimum visible radial height [Mm]")
        ax.set_title(observer)
        ax.grid(alpha=0.3)
    fig.suptitle("Original clicked points: conditional geometry, not confirmed flare roots", fontsize=12)
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.3)
    if output is not None:
        fig.savefig(output/"clicked_point_thresholds.png", dpi=160)
        plt.close(fig)
    return ranges_figure, fig


def calculate_assessment(config_path, settings=None):
    """Calculate hypotheses in memory. Optional settings do not modify JSON."""
    configure_cache()
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    config_path = config_path.resolve()
    assessment = json.loads(config_path.read_text()) if settings is None else settings
    selection_config = config_path.parent/assessment["selection_config"]
    config, selected, m, epoch, digest, messages = read_selection(selection_config)
    photosphere = m.rsun_meters
    stereo_observer = m.observer_coordinate
    observers, ephemeris = load_observers(epoch, config_path.parent/"ephemeris"/(epoch.utc.isot.replace(":", "-")+".json"))
    targets = {name: observers[name] for name in ("Solar Orbiter", "Earth")}
    point_rows = []
    for row in selected:
        source = source_coordinate(row, epoch)
        for name, observer in targets.items():
            result = radial_occultation(source, observer, photosphere)
            point_rows.append(dict(label=row["label"], observer=name, **{k: np.asarray(v).item() for k, v in result.items()}))
    scenarios = []
    grid_size = assessment["sample_grid_size"]
    if grid_size < 2:
        raise ValueError("Grid needs at least two samples per dimension")
    for region in assessment["regions"]:
        x0, x1, y0, y1 = region["bounds_arcsec"]
        xx, yy = np.meshgrid(np.linspace(x0, x1, grid_size), np.linspace(y0, y1, grid_size))
        for shift_x, shift_y in assessment["pointing_shift_tests_arcsec"]:
            for height_km in assessment["euv_height_tests_km"]:
                # If the selected EUV emission is at R+h, infer its radial root
                # from the same line of sight. Do not add h to the opaque radius.
                radius = photosphere + height_km*u.km
                direction = SkyCoord((xx.ravel()+shift_x)*u.arcsec, (yy.ravel()+shift_y)*u.arcsec,
                                     frame=frames.Helioprojective(observer=stereo_observer, obstime=epoch, rsun=radius))
                inferred = direction.make_3d().transform_to(frames.HeliographicStonyhurst(obstime=epoch))
                if not np.all(np.isfinite(inferred.radius.to_value(u.km))):
                    raise ValueError("Region includes undefined spherical intersections; change location hypothesis")
                root = SkyCoord(inferred.lon, inferred.lat, np.ones(xx.size)*photosphere,
                                frame=frames.HeliographicStonyhurst(obstime=epoch, rsun=photosphere))
                for name, observer in targets.items():
                    result = radial_occultation(root, observer, photosphere)
                    scenarios.append(dict(region=region["id"], observer=name, shift_x_arcsec=shift_x,
                                          shift_y_arcsec=shift_y, euv_height_assumed_km=height_km,
                                          sample_count=xx.size,
                                          root_lon_min_deg=float(inferred.lon.deg.min()), root_lon_max_deg=float(inferred.lon.deg.max()),
                                          root_lat_min_deg=float(inferred.lat.deg.min()), root_lat_max_deg=float(inferred.lat.deg.max()),
                                          theta_occ_min_deg=float(np.min(result["theta_occ_deg"])),
                                          theta_occ_max_deg=float(np.max(result["theta_occ_deg"])),
                                          h_occ_min_km=float(np.min(result["h_occ_km"])), h_occ_max_km=float(np.max(result["h_occ_km"])),
                                          fraction_surface_hidden=float(np.mean(result["surface_hidden"]))))
    summaries = []
    for region in assessment["regions"]:
        for name in targets:
            rows = [r for r in scenarios if r["region"] == region["id"] and r["observer"] == name]
            summaries.append(dict(region=region["id"], observer=name, scenario_count=len(rows),
                                  theta_occ_min_deg=min(r["theta_occ_min_deg"] for r in rows),
                                  theta_occ_max_deg=max(r["theta_occ_max_deg"] for r in rows),
                                  h_occ_min_km=min(r["h_occ_min_km"] for r in rows),
                                  h_occ_max_km=max(r["h_occ_max_km"] for r in rows),
                                  interpretation="Scenario envelope; not a confidence interval or confirmed occultation measurement"))
    report = dict(config=assessment, epoch_utc=epoch.utc.isot, fits_sha256=digest, ephemeris=ephemeris,
                  warnings=messages, formula="rcrit = R/[sqrt(1-(R/D)^2)*sin(alpha)+(R/D)*cos(alpha)]",
                  angle_conventions="theta_occ=alpha-90 deg; behind_finite_limb=alpha-acos(R/D); positive finite angle means hidden surface",
                  limitations=["Regions manually bracket morphology, not validated ribbon locations", "No photometric calibration or saturation mask", "Pointing shifts are illustrative sensitivity offsets, not measured uncertainties", "EUV heights 0/1/3 Mm are hypothetical, not inferred", "Radial-root and opaque spherical photosphere assumptions", "Region boxes at one epoch; contextual frames not feature-tracked", "No photon-emission-time alignment", "Threshold is not measured X-ray source height"])
    return dict(summaries=summaries, points=point_rows, scenarios=scenarios, report=report)


def run(config_path, output, context=True):
    result = calculate_assessment(config_path)
    output.mkdir(parents=True, exist_ok=True)
    if context:
        context_montage(config_path, output)
    write_table(output/"clicked_point_thresholds.csv", result["points"])
    write_table(output/"region_sensitivity.csv", result["scenarios"])
    write_table(output/"conditional_ranges.csv", result["summaries"])
    (output/"assessment.json").write_text(json.dumps(result["report"], indent=2))
    plot_ranges(result["summaries"], result["points"], output)
    print(f"Saved {output.resolve()}")
    for r in result["summaries"]:
        print(f"{r['region']} / {r['observer']}: conditional h_occ {r['h_occ_min_km']/1000:.3f}-{r['h_occ_max_km']/1000:.3f} Mm")
    return 0


def main(argv=None):
    configure_cache()
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-context", action="store_true", help="Recalculate thresholds without rebuilding context images")
    args = parser.parse_args(argv)
    config = args.config or default_geometry_config().parent/"location_assessment.json"
    return run(config, args.output or default_output(config, "location_assessment"), context=not args.skip_context)


if __name__ == "__main__":
    raise SystemExit(main())
