"""Geometry snapshot: observers, surface limbs and selected-point projections.

Uses the selected EUVI map's WCS reference epoch. This is not an emission-time
alignment or a height measurement. Surface locations remain provisional.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from occulted_flares.geometry.geometry_runtime import configure_cache, default_geometry_config, default_output
from occulted_flares.geometry.geometry_plotting import label_points
from occulted_flares.geometry.geometry_ephemeris import load_observers
from occulted_flares.geometry.coordinate_transforms import project_surface






def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_geometry(m, observers, selections, projections, output=None):
    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from astropy.visualization import ImageNormalize, AsinhStretch
    from sunpy.visualization.drawing import limb

    colors = {"Earth": "tab:blue", "Solar Orbiter": "tab:orange", "STEREO-A": "tab:green"}
    group_colors = {g: color for g, color in zip(dict.fromkeys(r["group"] for r in selections), ["cyan", "magenta", "yellow", "lime"])}
    fig = plt.figure(figsize=(19, 7))
    axp = fig.add_subplot(131, projection="polar")
    axp.scatter(0, 0, s=100, color="gold", edgecolor="black", label="Sun")
    for name, observer in observers.items():
        # True projection into the HGS equatorial plane, not an ecliptic plot.
        rho = observer.radius.to_value(u.AU)*np.cos(observer.lat.to_value(u.rad))
        theta = observer.lon.to_value(u.rad)
        axp.plot([theta, theta], [0, rho], color=colors[name], alpha=0.4)
        axp.scatter(theta, rho, color=colors[name], label=name)
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_ylim(0, 1.12*max(c.radius.to_value(u.AU) for c in observers.values()))
    axp.set_title("Observer positions\nHGS equatorial projection; radial units AU", fontsize=11)
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=9)

    axm = fig.add_subplot(132, projection=m)
    values = m.data[np.isfinite(m.data)]
    lo, hi = np.percentile(values, [1, 99.8])
    m.plot(axes=axm, norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch()))
    m.draw_limb(axes=axm, color="white", linewidth=1)
    for name in ("Earth", "Solar Orbiter"):
        limb(axm, observers[name], rsun=m.rsun_meters, color=colors[name], linewidth=1.5)
    for row in selections:
        axm.plot(float(row["x_pixel"]), float(row["y_pixel"]), "+", color=group_colors[row["group"]], markersize=7)
    handles = [Line2D([], [], color="white", label="STEREO photosphere (h = 0)"),
               Line2D([], [], color=colors["Earth"], label="Earth photosphere (h = 0)"),
               Line2D([], [], color=colors["Solar Orbiter"], label="Solar Orbiter photosphere (h = 0)")]
    axm.legend(handles=handles, loc="lower left", fontsize=8)
    axm.set_title(f"EUVI {m.wavelength}: photospheric boundaries, h = 0\nDotted boundary portions are behind the STEREO limb", fontsize=10)

    axs = fig.add_subplot(133)
    solo = observers["Solar Orbiter"]
    angular_radius = np.arcsin((m.rsun_meters/solo.radius).decompose().value)*u.rad.to(u.arcsec)
    axs.add_patch(Circle((0, 0), angular_radius, fill=False, color="gray"))
    for row in projections:
        if row["observer"] != "Solar Orbiter":
            continue
        axs.plot(row["tx_arcsec"], row["ty_arcsec"], "o" if row["surface_visible"] else "x", color=group_colors[row["group"]])
    axs.set_xlim(-1.12*angular_radius, 1.12*angular_radius)
    axs.set_ylim(-1.12*angular_radius, 1.12*angular_radius)
    axs.set_aspect("equal")
    axs.grid(alpha=0.3)
    axs.set_xlabel("Solar X [arcsec]")
    axs.set_ylabel("Solar Y [arcsec]")
    axs.set_title("Selected surface points from Solar Orbiter\nx = occulted; circle = visible; not a STIX image", fontsize=10)
    axs.legend(handles=[Line2D([], [], marker="+", linestyle="none", color=c, label=g.replace("_", " ")) for g, c in group_colors.items()], loc="lower left", fontsize=8)
    fig.suptitle(f"Geometry snapshot at {m.reference_date.utc.isot} UTC (EUVI WCS epoch)", fontsize=13)
    fig.text(0.5, 0.02, "Provisional photospheric locations; no height measurement or photon-emission-time alignment. Longitudes/latitudes are retained in 3D calculations.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.16, wspace=0.42)
    if output is not None:
        fig.savefig(output / "observer_geometry.png", dpi=160)
        plt.close(fig)

    # Separate close-up keeps the full-disk overview legible. Label positions
    # move; measured positions and limb coordinates do not.
    detail = plt.figure(figsize=(14, 7))
    left = detail.add_subplot(121, projection=m)
    xs = np.array([float(r["x_pixel"]) for r in selections])
    ys = np.array([float(r["y_pixel"]) for r in selections])
    pad = max(65, 0.5*max(np.ptp(xs), np.ptp(ys)))
    crop = m.data[max(0, int(ys.min()-pad)):min(m.data.shape[0], int(ys.max()+pad)+1),
                  max(0, int(xs.min()-pad)):min(m.data.shape[1], int(xs.max()+pad)+1)]
    detail_lo, detail_hi = np.percentile(crop[np.isfinite(crop)], [1, 99.9])
    m.plot(axes=left, norm=ImageNormalize(vmin=detail_lo, vmax=max(detail_hi, detail_lo+1), stretch=AsinhStretch()))
    left.set_xlim(xs.min()-pad, xs.max()+pad)
    left.set_ylim(ys.min()-pad, ys.max()+pad)
    limb(left, observers["Solar Orbiter"], rsun=m.rsun_meters, color="tab:orange", linewidth=1.5)
    for r in selections:
        left.plot(float(r["x_pixel"]), float(r["y_pixel"]), "+", color=group_colors[r["group"]], markersize=9)
    label_points(left, [(float(r["x_pixel"]), float(r["y_pixel"]), r["label"]) for r in selections])
    left.set_title("EUVI candidate close-up\nOrange = Solar Orbiter photospheric limb (h = 0)", fontsize=11)
    right = detail.add_subplot(122)
    chosen = [r for r in projections if r["observer"] == "Solar Orbiter"]
    px = np.array([r["tx_arcsec"] for r in chosen])
    py = np.array([r["ty_arcsec"] for r in chosen])
    span = max(np.ptp(px), np.ptp(py), 40)
    right.add_patch(Circle((0, 0), angular_radius, fill=False, color="gray"))
    right.set_xlim(px.mean()-0.85*span, px.mean()+0.85*span)
    right.set_ylim(py.mean()-0.85*span, py.mean()+0.85*span)
    for r in chosen:
        right.plot(r["tx_arcsec"], r["ty_arcsec"], "o" if r["surface_visible"] else "x", color="tab:blue", markersize=8)
    label_points(right, [(r["tx_arcsec"], r["ty_arcsec"], r["label"]) for r in chosen], color="tab:blue")
    right.set_aspect("equal")
    right.grid(alpha=0.3)
    right.set_xlabel("Solar X [arcsec]")
    right.set_ylabel("Solar Y [arcsec]")
    right.set_title("Projection from Solar Orbiter\nx = occulted surface; o = visible surface", fontsize=11)
    detail.suptitle(f"Selected point IDs | {m.reference_date.utc.isot} UTC", fontsize=12)
    detail.text(0.5, 0.025, "Surface assumption; candidate association unconfirmed. Leader lines identify points without shifting coordinates.", ha="center", fontsize=9)
    detail.subplots_adjust(top=0.84, bottom=0.16, wspace=0.4)
    if output is not None:
        detail.savefig(output / "observer_geometry_closeup.png", dpi=170)
        plt.close(detail)
    return fig, detail


def run(config_path, output, refresh=False):
    configure_cache()
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    from occulted_flares.geometry.euvi_coordinates import load_map

    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    selection_run = base / config["selection_run"]
    with (selection_run / "coordinates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["file"] for r in rows}) != 1:
        raise ValueError("Select points from one map per geometry snapshot")
    membership = {label: name for name, labels in config["groups"].items() for label in labels}
    if len(membership) != sum(map(len, config["groups"].values())):
        raise ValueError("A point belongs to more than one group")
    if set(membership) - {r["label"] for r in rows}:
        raise ValueError("Configured group references unknown point labels")
    selected = [dict(r, group=membership[r["label"]]) for r in rows if r["label"] in membership]
    if not selected or any(r["status"] != "surface_assumed_not_height_measured" for r in selected):
        raise ValueError("Surface geometry requires valid surface-assumed selections")
    fits_path = base / config["fits_directory"] / rows[0]["file"]
    with fits_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = json.loads((selection_run / "metadata.json").read_text())["files"][0]["sha256"]
    if digest != expected:
        raise ValueError("FITS checksum differs from selection provenance")
    m, header_observer, epoch, warnings = load_map(fits_path, False)
    if any(r["wcs_reference_time"] != epoch.isot for r in selected):
        raise ValueError("Selection epochs do not match WCS reference time")
    cache = base / "ephemeris" / (epoch.utc.isot.replace(":", "-") + ".json")
    observers, provenance = load_observers(epoch, cache, refresh)
    # Compare independent Horizons and FITS header coordinates, without replacing either.
    delta = (header_observer.cartesian.xyz - observers["STEREO-A"].cartesian.xyz).to_value(u.km)
    offset = float(np.linalg.norm(delta))
    projected = []
    for row in selected:
        source = SkyCoord(float(row["hgs_lon_deg"])*u.deg, float(row["hgs_lat_deg"])*u.deg,
                          float(row["assumed_radius_km"])*u.km, frame=frames.HeliographicStonyhurst(obstime=epoch))
        for name, observer in observers.items():
            projected.append(dict(group=row["group"], label=row["label"], observer=name,
                                  epoch_utc=epoch.utc.isot, **project_surface(source, observer)))
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "projected_points.csv", projected)
    write_table(output / "observer_positions.csv", [dict(observer=name, epoch_utc=epoch.utc.isot, **p) for name, p in provenance["observers"].items()])
    report = dict(epoch_utc=epoch.utc.isot, input_sha256=digest, config=config, ephemeris=provenance,
                  stereo_header_horizons_offset_km=offset, warnings=warnings,
                  assumptions=["Snapshot at EUVI WCS reference epoch, not time-aligned photon emission", "Photospheric source locations", "No differential rotation or source evolution model", "Earth center approximates Earth-based observers", "Projected occulted points are directions, not visible emission"],
                  next_checks=["Investigate header/ephemeris differences", "Confirm candidate low-atmosphere identity", "Compare with calibrated STIX astrometry", "Emission times and height uncertainty in later stage"])
    (output / "geometry.json").write_text(json.dumps(report, indent=2))
    plot_geometry(m, observers, selected, projected, output)
    print(f"Saved {output.resolve()}; STEREO header/Horizons position offset: {offset:.1f} km")
    return 0


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Defaults to the event in default_event.json")
    p.add_argument("--output", type=Path, help="Defaults to outputs/observers; generated files are replaced")
    p.add_argument("--refresh", action="store_true", help="Refresh Horizons cache for this map epoch")
    args = p.parse_args(argv)
    args.config = (args.config or default_geometry_config()).resolve()
    args.output = args.output or default_output(args.config, "observers")
    return args


def main(argv=None):
    configure_cache()
    import matplotlib
    matplotlib.use("Agg")  # CLI only; reusable plots retain notebook backend.
    args = arguments(argv)
    print(f"Configuration: {args.config}\nOutput: {args.output}", flush=True)
    return run(args.config, args.output, args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
