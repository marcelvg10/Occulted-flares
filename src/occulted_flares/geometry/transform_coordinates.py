"""Transform selected surface-assumed points to three observer viewpoints.

Run directly in VS Code. Defaults come from default_event.json; generated files
replace Geometry/outputs/coordinate_transforms. No source motion, height fitting
or common photon-emission-time alignment is performed.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

from occulted_flares.geometry.geometry_runtime import configure_cache, default_geometry_config, default_output
from occulted_flares.geometry.coordinate_transforms import source_coordinate, transform_with_check
from occulted_flares.geometry.geometry_ephemeris import load_observers


def read_selection(config_path):
    """Load current selected groups and verify their FITS and epoch provenance."""
    from occulted_flares.geometry.euvi_coordinates import load_map
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    selection = base / config["selection_run"]
    with (selection / "coordinates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {label: group for group, labels in config["groups"].items() for label in labels}
    if not groups or len(groups) != sum(len(v) for v in config["groups"].values()):
        raise ValueError("Need nonempty groups with unique point labels")
    if not rows or set(groups)-{r["label"] for r in rows}:
        raise ValueError("Configured labels are absent from the current selection")
    selected = [dict(r, group=groups[r["label"]]) for r in rows if r["label"] in groups]
    if len({r["file"] for r in selected}) != 1:
        raise ValueError("Use selections from one image per snapshot")
    if any(r["status"] != "surface_assumed_not_height_measured" for r in selected):
        raise ValueError("Input requires valid surface-assumed heliographic coordinates")
    fits_path = base / config["fits_directory"] / selected[0]["file"]
    with fits_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    stored = json.loads((selection / "metadata.json").read_text())
    matching = [r for r in stored["files"] if Path(r["path"]).name == fits_path.name]
    if len(matching) != 1 or digest != matching[0]["sha256"]:
        raise ValueError("Input FITS does not match selection metadata")
    m, _, epoch, messages = load_map(fits_path, False)
    if any(r["wcs_reference_time"] != epoch.isot for r in selected):
        raise ValueError("Selection and image coordinate epochs differ")
    return config, selected, m, epoch, digest, messages


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(projections, observers, radius, output=None):
    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from occulted_flares.geometry.geometry_plotting import label_points

    fig, axes = plt.subplots(1, len(observers), figsize=(17, 6))
    for ax, (name, observer) in zip(axes, observers.items()):
        chosen = [r for r in projections if r["observer"] == name]
        xs = np.array([r["tx_arcsec"] for r in chosen])
        ys = np.array([r["ty_arcsec"] for r in chosen])
        angular_radius = np.arcsin((radius/observer.radius).decompose().value)*u.rad.to(u.arcsec)
        ax.add_patch(Circle((0, 0), angular_radius, fill=False, color="gray", linewidth=1.5))
        span = max(np.ptp(xs), np.ptp(ys), 50)
        ax.set_xlim(xs.mean()-span*0.85, xs.mean()+span*0.85)
        ax.set_ylim(ys.mean()-span*0.85, ys.mean()+span*0.85)
        for r in chosen:
            visible = r["surface_visible"]
            ax.plot(r["tx_arcsec"], r["ty_arcsec"], "o" if visible else "x",
                    color="tab:blue" if visible else "tab:red", markersize=7)
        label_points(ax, [(r["tx_arcsec"], r["ty_arcsec"], r["label"]) for r in chosen], color="black")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Solar X [arcsec]")
        ax.set_ylabel("Solar Y [arcsec]")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal")
    handles = [Line2D([], [], color="gray", label="Photospheric limb"),
               Line2D([], [], marker="o", linestyle="none", color="tab:blue", label="Visible surface"),
               Line2D([], [], marker="x", linestyle="none", color="tab:red", label="Hidden surface: projected direction only")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=9)
    fig.suptitle(f"Same selected surface points in three viewpoints | {projections[0]['epoch_utc']} UTC\nAngular-coordinate zooms, not telescope images; panel limits differ", fontsize=12)
    fig.subplots_adjust(top=0.81, bottom=0.22, wspace=0.4)
    if output is not None:
        fig.savefig(output / "coordinate_comparison.png", dpi=170)
        plt.close(fig)
    return fig


def calculate_transforms(config_path):
    """Return fresh calculations in memory; write only a missing ephemeris cache."""
    configure_cache()
    import numpy as np
    import astropy.units as u
    config, selected, m, epoch, digest, warnings = read_selection(config_path)
    cache_path = config_path.resolve().parent / "ephemeris" / (epoch.utc.isot.replace(":", "-") + ".json")
    observers, ephemeris = load_observers(epoch, cache_path)
    projections, source_rows, checks = [], [], []
    for row in selected:
        source = source_coordinate(row, epoch)
        # This check uses the FITS observer (not Horizons), so it isolates the
        # pixel->surface->pixel conversion from independent ephemeris differences.
        back_direction = source.transform_to(m.coordinate_frame)
        back_pixel = m.world_to_pixel(back_direction)
        pixel_error = float(np.hypot(back_pixel.x.to_value(u.pix)-float(row["x_pixel"]),
                                    back_pixel.y.to_value(u.pix)-float(row["y_pixel"])))
        checks.append(dict(label=row["label"], source_pixel_roundtrip_error_pix=pixel_error))
        source_rows.append({key: row[key] for key in ("label", "group", "file", "wcs_reference_time", "x_pixel", "y_pixel", "tx_arcsec", "ty_arcsec", "hgs_lon_deg", "hgs_lat_deg", "assumed_radius_km")})
        for name, observer in observers.items():
            projections.append(dict(label=row["label"], group=row["group"], observer=name,
                                    epoch_utc=epoch.utc.isot, **transform_with_check(source, observer)))
    max_pixel_error = max(r["source_pixel_roundtrip_error_pix"] for r in checks)
    max_3d_error = max(r["roundtrip_3d_error_m"] for r in projections)
    # Numerical consistency tolerances, not observational error estimates.
    if max_pixel_error > 1e-4 or max_3d_error > 1:
        raise ValueError(f"Transformation consistency failed: {max_pixel_error} pix / {max_3d_error} m")
    manifest = dict(epoch_utc=epoch.utc.isot, config=config, source_fits_sha256=digest,
                    versions={p: importlib.metadata.version(p) for p in ("sunpy", "astropy", "numpy")},
                    ephemeris=ephemeris, warnings=warnings,
                    max_pixel_roundtrip_error_pix=max_pixel_error, max_3d_roundtrip_error_m=max_3d_error,
                    assumptions=["All sources lie on the adopted photosphere", "HGS frame and epoch unchanged between viewpoints", "No source evolution, rotation to another time or light-travel alignment", "Hidden points may still have finite angular coordinates", "Numerical residuals are not physical position uncertainties", "Solar Orbiter helioprojective coordinates are not raw STIX instrument-frame coordinates"])
    return dict(map=m, selected=selected, observers=observers, projections=projections,
                sources=source_rows, checks=checks, manifest=manifest)


def run(config_path, output):
    result = calculate_transforms(config_path)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "source_coordinates.csv", result["sources"])
    write_table(output / "transformed_coordinates.csv", result["projections"])
    write_table(output / "validation.csv", result["checks"])
    (output / "transformations.json").write_text(json.dumps(result["manifest"], indent=2))
    plot_comparison(result["projections"], result["observers"], result["map"].rsun_meters, output)
    print(f"Saved {output.resolve()}")
    max_pixel_error = result["manifest"]["max_pixel_roundtrip_error_pix"]
    max_3d_error = result["manifest"]["max_3d_roundtrip_error_m"]
    print(f"Numerical checks: {max_pixel_error:.3g} pixel; {max_3d_error:.3g} metre (not physical accuracy).")
    for name in result["observers"]:
        visible = sum(r["surface_visible"] for r in result["projections"] if r["observer"] == name)
        print(f"{name}: {visible}/{len(result['selected'])} assumed surface points visible")
    return 0


def main(argv=None):
    configure_cache()
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = (args.config or default_geometry_config()).resolve()
    output = args.output or default_output(config, "coordinate_transforms")
    return run(config, output)


if __name__ == "__main__":
    raise SystemExit(main())
