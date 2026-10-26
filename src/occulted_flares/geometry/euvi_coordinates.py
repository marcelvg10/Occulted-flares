"""Inspect EUVI FITS and measure selected points; no calibration or ephemeris query.

Run --help first. Pixel positions refer to the original FITS array, zero based,
with x increasing along columns and y along rows (display origin lower).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings
from occulted_flares.geometry.geometry_runtime import configure_cache, configure_gui


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="FITS files or directories (recursive)")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; refuses existing directory")
    parser.add_argument("--points", type=Path, help="CSV: file,label,x_pixel,y_pixel; file is exact basename")
    parser.add_argument("--pick", action="store_true", help="Click points, then press Enter; one image at a time")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in an existing output folder")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow generic maps for synthetic tests only")
    return parser.parse_args(argv)


def discover(inputs):
    suffixes = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
    found = set()
    for entry in inputs:
        if not entry.exists():
            raise ValueError(f"Input does not exist: {entry}")
        candidates = entry.rglob("*") if entry.is_dir() else [entry]
        found.update(p.resolve() for p in candidates if p.is_file() and p.name.lower().endswith(suffixes))
    if not found:
        raise ValueError("No FITS files found. JPEG browse images cannot supply reliable FITS coordinates.")
    files = sorted(found)
    if len({p.name for p in files}) != len(files):
        raise ValueError("Duplicate FITS basenames: select one version/location per file.")
    return files


def read_points(path, files):
    points = {p.name: [] for p in files}
    if path:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"file", "label", "x_pixel", "y_pixel"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Point CSV needs columns {sorted(required)}")
            for row in reader:
                if row["file"] not in points:
                    raise ValueError(f"Point file not in inputs: {row['file']}")
                points[row["file"]].append((row["label"], float(row["x_pixel"]), float(row["y_pixel"])))
    return points


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_map(path, allow_synthetic):
    import astropy.units as u
    import sunpy.map
    from sunpy.coordinates import frames
    from sunpy.map.sources import EUVIMap

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = sunpy.map.Map(path)
        if not allow_synthetic and not isinstance(m, EUVIMap):
            raise ValueError("Expected a STEREO EUVI map; check instrument/product.")
        # Do not silently accept a default observation time or Earth observer.
        if not any(m.meta.get(key) for key in ("date-obs", "date_obs", "date-beg")):
            raise ValueError("Missing observation date")
        observer = m.observer_coordinate
        reference_date = m.reference_date
        if not isinstance(m.coordinate_frame, frames.Helioprojective):
            raise ValueError("Expected helioprojective WCS")
        m.pixel_to_world(0 * u.pix, 0 * u.pix)
    messages = [str(w.message) for w in caught]
    if any("observer" in message.lower() and ("missing" in message.lower() or "assuming" in message.lower()) for message in messages):
        raise ValueError("Observer metadata incomplete: " + "; ".join(messages))
    return m, observer, reference_date, messages


def measure(m, label, x, y):
    import numpy as np
    import astropy.units as u
    from sunpy.coordinates import frames

    result = dict(label=label, x_pixel=x, y_pixel=y, tx_arcsec="", ty_arcsec="",
                  hgs_lon_deg="", hgs_lat_deg="", assumed_radius_km=m.rsun_meters.to_value(u.km),
                  status="outside_image")
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= m.data.shape[1]-1 and 0 <= y <= m.data.shape[0]-1):
        return result
    direction = m.pixel_to_world(x * u.pix, y * u.pix)
    result.update(tx_arcsec=direction.Tx.to_value(u.arcsec), ty_arcsec=direction.Ty.to_value(u.arcsec))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Off-limb intersection is explicitly flagged below.
        surface = direction.make_3d()
    if not np.isfinite(surface.distance.to_value(u.km)):
        result["status"] = "off_limb_surface_undefined"
        return result
    hgs = surface.transform_to(frames.HeliographicStonyhurst(obstime=m.reference_date))
    result.update(hgs_lon_deg=hgs.lon.to_value(u.deg), hgs_lat_deg=hgs.lat.to_value(u.deg),
                  status="surface_assumed_not_height_measured")
    return result


def main(argv=None):
    args = arguments(argv)
    files = discover(args.inputs)  # Fail before imports/output if only JPEGs exist.
    points = read_points(args.points, files)
    configure_cache()
    if args.pick:
        configure_gui()
    import matplotlib
    if not args.pick:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, AsinhStretch
    import numpy as np

    if args.pick and str(matplotlib.get_backend()).lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        raise ValueError("Interactive selection needs a GUI backend (for example TkAgg). Use --points for saved pixel coordinates instead.")

    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    report = dict(purpose="Exploratory EUVI coordinates; no calibration, registration or light-time correction",
                  synthetic_test=args.allow_synthetic, pixel_convention="original FITS zero-based x=column, y=row",
                  coordinate_assumption="on-disk points intersect photosphere; off-limb undefined",
                  uncertainty="No automatic uncertainty estimate; select extent/control points and assess pointing/height separately",
                  versions={p: importlib.metadata.version(p) for p in ("sunpy", "astropy", "matplotlib", "numpy")}, files=[])
    coordinates, selected = [], []
    failures = 0
    for index, path in enumerate(files):
        entry = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["files"].append(entry)
        try:
            m, observer, date, messages = load_map(path, args.allow_synthetic)
            values = np.asarray(m.data, dtype=float)
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError("No finite image values")
            lo, hi = np.percentile(finite, [1, 99.8])
            if hi <= lo:
                hi = lo + 1
            entry.update(date_obs=m.date.isot, wcs_reference_time=date.isot,
                         wavelength=str(m.wavelength), observer_hgs=dict(lon_deg=observer.lon.to_value("deg"),
                         lat_deg=observer.lat.to_value("deg"), radius_m=observer.radius.to_value("m")),
                         shape=list(values.shape), warnings=messages,
                         header={key: str(m.meta.get(key, "")) for key in ("telescop", "instrume", "obsrvtry", "exptime", "level", "bunit", "rsun_ref", "rsun_obs", "crota", "crota2", "history")},
                         finite_max=float(finite.max()), pixels_at_max=int(np.count_nonzero(values == finite.max())),
                         display_limits=[float(lo), float(hi)],
                         assumed_radius_km=m.rsun_meters.to_value("km"),
                         radius_source="FITS RSUN_REF" if m.meta.get("rsun_ref") is not None else "SunPy inferred/default radius; review RSUN_OBS/DSUN_OBS if present",
                         review_required=["Processing/calibration level", "Pointing and saturation", "Surface assumption", "Exposure timestamps and time convention"])
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection=m)
            m.plot(axes=ax, norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch()))
            m.draw_limb(axes=ax, color="white", linewidth=0.8)
            ax.set_title(f"{path.name}\n{m.date.isot} | {m.wavelength}\nExploratory display; saturation/processing unverified", fontsize=9)
            fig.text(0.5, 0.02, "Points use original FITS pixels. Heliographic conversion assumes the solar surface.", ha="center", fontsize=8)
            picked = list(points[path.name])
            if args.pick:
                print(f"{path.name}: click feature/extent points; Enter finishes. Avoid streaks and elevated loops.")
                fig.canvas.draw()
                picked.extend((f"point_{j+1}", x, y) for j, (x, y) in enumerate(plt.ginput(n=-1, timeout=0)))
            for label, x, y in picked:
                row = measure(m, label, x, y)
                coordinates.append(dict(file=path.name, observation_time=m.date.isot, wcs_reference_time=date.isot, **row))
                selected.append(dict(file=path.name, label=label, x_pixel=x, y_pixel=y))
                if row["status"] != "outside_image":
                    ax.plot(x, y, marker="+", color="cyan", linestyle="none")
                    ax.text(x+5, y+5, label, color="cyan", fontsize=8)
            fig.savefig(args.output / f"{index+1:03d}_{path.name}.png", dpi=150)
            plt.close(fig)
            entry["status"] = "plotted"
        except Exception as error:
            failures += 1
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            plt.close("all")
            print(f"FAILED {path.name}: {error}")
    write_csv(args.output / "selected_pixels.csv", selected, ["file", "label", "x_pixel", "y_pixel"])
    write_csv(args.output / "coordinates.csv", coordinates, ["file", "observation_time", "wcs_reference_time", "label", "x_pixel", "y_pixel", "tx_arcsec", "ty_arcsec", "hgs_lon_deg", "hgs_lat_deg", "assumed_radius_km", "status"])
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output.resolve()}; {len(files)-failures}/{len(files)} images processed.")
    print(f"Selected {len(selected)} points. Coordinates describe only these selections.")
    if not selected:
        print("No points selected: CSV files contain headers only. Use --pick or --points to measure coordinates.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, ImportError) as error:
        raise SystemExit(str(error)) from error
