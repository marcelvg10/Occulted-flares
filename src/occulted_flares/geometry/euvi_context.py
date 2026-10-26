"""Qualitative matched EUVI crops: common angular bounds and per-band stretch.

No difference photometry, calibration or solar-rotation tracking is implied.
"""
import csv
import hashlib
import json
from pathlib import Path


def context_montage(config_path, output=None):
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.visualization import ImageNormalize, AsinhStretch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from occulted_flares.geometry.euvi_coordinates import load_map

    config = json.loads(config_path.read_text())
    selection_config = json.loads((config_path.parent/config["selection_config"]).read_text())
    fits_dir = config_path.parent/selection_config["fits_directory"]
    xmin, xmax, ymin, ymax = config["context_bounds_arcsec"]
    fig = plt.figure(figsize=(17, 10))
    checks = []
    for row_index, (wave, filenames) in enumerate(config["context_frames"].items()):
        maps = []
        for filename in filenames:
            path = fits_dir/filename
            m, observer, epoch, messages = load_map(path, False)
            if abs(m.wavelength.to_value(u.angstrom)-float(wave)) > 0.01:
                raise ValueError("Unexpected wavelength")
            bottom_left = SkyCoord(xmin*u.arcsec, ymin*u.arcsec, frame=m.coordinate_frame)
            top_right = SkyCoord(xmax*u.arcsec, ymax*u.arcsec, frame=m.coordinate_frame)
            crop = m.submap(bottom_left, top_right=top_right)
            maps.append(crop)
            data = np.asarray(crop.data, dtype=float)
            checks.append(dict(file=filename, wavelength_angstrom=wave, date_obs=m.date.isot,
                               reference_time=epoch.isot, exposure_s=m.exposure_time.to_value(u.s),
                               units=str(m.unit), crop_min=float(np.nanmin(data)), crop_max=float(np.nanmax(data)),
                               high_dn_ge16000_pixels=int(np.count_nonzero(data>=16000)),
                               high_dn_flag_meaning="diagnostic only; not a calibrated saturation mask",
                               processing_level=str(m.meta.get("level", "unknown")),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        joint = np.concatenate([m.data[np.isfinite(m.data)].ravel() for m in maps])
        lo, hi = np.percentile(joint, [1, 99.8])
        for column, m in enumerate(maps):
            ax = fig.add_subplot(len(config["context_frames"]), len(filenames), row_index*len(filenames)+column+1, projection=m)
            m.plot(axes=ax, norm=ImageNormalize(vmin=lo, vmax=max(hi, lo+1), stretch=AsinhStretch()))
            ax.set_title(f"{wave} A | {m.date.isot[11:19]} UTC\n{m.exposure_time.to_value(u.s):.2f} s exposure", fontsize=10)
            if row_index == 0:
                ax.set_xlabel("")
            for region in config.get("regions", []):
                x0, x1, y0, y1 = region["bounds_arcsec"]
                bottom = SkyCoord(x0*u.arcsec, y0*u.arcsec, frame=m.coordinate_frame)
                top = SkyCoord(x1*u.arcsec, y1*u.arcsec, frame=m.coordinate_frame)
                px0, py0 = m.world_to_pixel(bottom)
                px1, py1 = m.world_to_pixel(top)
                ax.add_patch(Rectangle((px0.value, py0.value), px1.value-px0.value, py1.value-py0.value,
                                       fill=False, color=region.get("color", "cyan"), lw=1.2))
                ax.text(px0.value, py0.value-3, region["id"], color=region.get("color", "cyan"), fontsize=8)
    fig.suptitle("Upper candidate: before / bright phase / after\nSame angular field and stretch within each wavelength; observer-arrival UTC", fontsize=13)
    fig.text(0.5, 0.025, "Original DN display: photometry/saturation not calibrated. No feature tracking or emission-time alignment. Boxes are hypotheses, not measured ribbons.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.12, hspace=0.4, wspace=0.3)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/"context_sequence.png", dpi=160)
        with (output/"context_checks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
        (output/"context_settings.json").write_text(json.dumps(config, indent=2))
        plt.close(fig)
    return fig, checks


if __name__ == "__main__":
    from occulted_flares.geometry.geometry_runtime import configure_cache, default_geometry_config, default_output
    configure_cache()
    import matplotlib
    matplotlib.use("Agg")
    config = default_geometry_config().parent/"location_assessment.json"
    context_montage(config, default_output(default_geometry_config(), "location_assessment"))
