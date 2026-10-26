"""Plot and summarize explicitly grouped EUVI selections; no flare identification.

Extrema describe sampled points, not ribbon boundaries or confidence intervals.
All input files remain unchanged. Generated outputs are replaced on each run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from occulted_flares.geometry.geometry_runtime import configure_cache, default_geometry_config, default_output
from occulted_flares.geometry.geometry_plotting import label_points


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Defaults to the event in default_event.json")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/selection_review; generated files are replaced")
    args = parser.parse_args(argv)
    args.config = (args.config or default_geometry_config()).resolve()
    args.output = args.output or default_output(args.config, "selection_review")
    return args


def review_selection(config_path, output=None):
    """Build a fresh review figure and table; save only when output is supplied."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    selection = config_path.parent / config["selection_run"]
    with (selection / "coordinates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["file"] for r in rows}) != 1:
        raise ValueError("This review expects one selected FITS image")
    labels = {r["label"] for r in rows}
    for name, group in config["groups"].items():
        if not group or not set(group).issubset(labels):
            raise ValueError(f"Invalid selection group: {name}")
    fits_path = config_path.parent / config["fits_directory"] / rows[0]["file"]
    original = json.loads((selection / "metadata.json").read_text())["files"][0]
    with fits_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != original["sha256"]:
        raise ValueError("Selected FITS checksum differs from selection run")
    configure_cache()
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import ImageNormalize, AsinhStretch
    from occulted_flares.geometry.euvi_coordinates import load_map

    m, observer, date, messages = load_map(fits_path, False)
    fig = plt.figure(figsize=(8*len(config["groups"]), 7.5))
    summaries = []
    for index, (name, members) in enumerate(config["groups"].items(), 1):
        chosen = [r for r in rows if r["label"] in members]
        if any(r["status"] != "surface_assumed_not_height_measured" for r in chosen):
            raise ValueError("Group includes undefined surface coordinates")
        xs = np.array([float(r["x_pixel"]) for r in chosen])
        ys = np.array([float(r["y_pixel"]) for r in chosen])
        pad = max(65, 0.4*max(np.ptp(xs), np.ptp(ys)))
        xmin, xmax = max(0, xs.min()-pad), min(m.data.shape[1]-1, xs.max()+pad)
        ymin, ymax = max(0, ys.min()-pad), min(m.data.shape[0]-1, ys.max()+pad)
        crop = m.data[int(ymin):int(ymax)+1, int(xmin):int(xmax)+1]
        finite = crop[np.isfinite(crop)]
        lo, hi = np.percentile(finite, [1, 99.9])
        ax = fig.add_subplot(1, len(config["groups"]), index, projection=m)
        m.plot(axes=ax, norm=ImageNormalize(vmin=lo, vmax=max(hi, lo+1), stretch=AsinhStretch()))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(name.replace("_", " "), fontsize=11)
        for r in chosen:
            x, y = float(r["x_pixel"]), float(r["y_pixel"])
            ax.plot(x, y, "+", color="cyan", markersize=9)
        label_points(ax, [(float(r["x_pixel"]), float(r["y_pixel"]), r["label"]) for r in chosen])
        lon = [float(r["hgs_lon_deg"]) for r in chosen]
        lat = [float(r["hgs_lat_deg"]) for r in chosen]
        summaries.append(dict(group=name, points=";".join(members), count=len(chosen),
                              hgs_lon_min_deg=min(lon), hgs_lon_max_deg=max(lon),
                              hgs_lat_min_deg=min(lat), hgs_lat_max_deg=max(lat)))
    fig.suptitle(f"EUVI {m.wavelength} | {m.date.isot}\nCandidate selections; association and surface-feature identity unconfirmed", fontsize=11)
    fig.text(0.5, 0.025, "Each panel has its own display stretch. Point extent is not a ribbon boundary or an uncertainty estimate.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.17, wspace=0.35)
    provenance = dict(config=config, input_sha256=digest, fits_path=str(fits_path.resolve()),
                      wcs_reference_time=date.isot, warnings=messages,
                      interpretation="Surface-assumed sampled locations only; neither flare confirmation nor confidence intervals")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "candidate_closeups.png", dpi=160)
        plt.close(fig)
        with (output / "candidate_ranges.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        (output / "review.json").write_text(json.dumps(provenance, indent=2))
    return dict(figure=fig, summaries=summaries, provenance=provenance)


def main(argv=None):
    configure_cache()
    import matplotlib
    matplotlib.use("Agg")
    args = arguments(argv)
    review_selection(args.config, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
