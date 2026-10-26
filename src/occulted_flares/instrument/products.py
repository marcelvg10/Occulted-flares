"""Read native STIX L1 products without applying scientific corrections.

Astropy applies the FITS column scaling. Column units, rather than catalogue
cadence or a hard-coded centisecond multiplier, determine elapsed seconds.
Housekeeping TIMEDEL=0 denotes point samples here, not continuous exposure.
"""
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy as np
import pandas as pd


def time_seconds(hdu, column):
    """Convert a scaled FITS time column to seconds; reject missing units."""
    unit = hdu.columns[column].unit
    if not unit:
        raise ValueError(f"Missing time unit for {column}")
    values = np.asarray(hdu.data[column], dtype=float) * u.Unit(unit)
    seconds = values.to_value(u.s)
    if not np.isfinite(seconds).all():
        raise ValueError(f"Non-finite {column}")
    return seconds


def read_product(path):
    """Copy small state fields and energy-summed counts; retain native times.

    CPD array dimensions are checked against the explicit ENERGY table and
    detector/pixel metadata. Every telemetered detector/pixel is included in
    this initial diagnostic; it is not a recommended analysis selection.
    """
    path = Path(path).resolve()
    with fits.open(path) as hdus:
        header = hdus[0].header.copy()
        if header.get("LEVEL") != "L1" or header.get("INSTRUME") != "STIX":
            raise ValueError("Expected STIX L1 product")
        if header.get("TIMESYS", "").strip().upper() != "UTC":
            raise ValueError("Explicit UTC time system required")
        data = hdus["DATA"]
        offsets = time_seconds(data, "time")
        durations = time_seconds(data, "timedel")
        if len(offsets) == 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("Non-empty, strictly increasing native times required")
        time = Time(header["DATE-BEG"], scale="utc") + TimeDelta(offsets, format="sec")
        result = dict(path=path, header=header, time=time, offset_s=offsets,
                      duration_s=durations,
                      columns=pd.DataFrame([dict(column=c.name, format=c.format,
                          unit=c.unit or "", dimensions=c.dim or "") for c in data.columns]),
                      extensions=[h.name for h in hdus],
                      idb_versions=hdus["IDB_VERSIONS"].data["version"].tolist())
        if "counts" in data.columns.names:
            if np.any(durations <= 0):
                raise ValueError("CPD integrations must be positive")
            energies = hdus["ENERGIES"].data
            counts = data.data["counts"]
            detector_mask = np.asarray(data.data["detector_masks"]).copy()
            pixel_mask = np.asarray(data.data["pixel_masks"]).copy()
            pixel_sets = np.asarray(data.data["num_pixel_sets"]).copy()
            if (counts.ndim != 4 or counts.shape[1] != detector_mask.shape[1]
                    or counts.shape[-1] != len(energies)
                    or not np.all(pixel_sets == counts.shape[2])):
                raise ValueError("Unsupported CPD dimensions: inspect before summing")
            result.update(kind="cpd", counts_shape=counts.shape,
                counts_by_energy=np.sum(counts, axis=(1, 2), dtype=float),
                energies=pd.DataFrame({n: np.asarray(energies[n]).astype(float) for n in energies.names}),
                states={"rcr": np.asarray(data.data["rcr"]).copy(),
                        "detector_masks": detector_mask, "pixel_masks": pixel_mask,
                        "num_pixel_sets": pixel_sets},
                start=time-TimeDelta(durations/2, format="sec"),
                end=time+TimeDelta(durations/2, format="sec"))
        else:
            if "instrument_mode" not in data.columns.names or np.any(durations != 0):
                raise ValueError("Expected HK Maxi point samples with TIMEDEL=0")
            result.update(kind="hk", states={n: np.asarray(data.data[n]).copy()
                for n in data.columns.names if n not in ("time", "timedel", "control_index")})
    return result


def band_rate(product, low, high):
    """Uncorrected counts/native integration second in complete energy bins.

    No energy-width/area/distance/livetime/ELUT correction or subtraction.
    Reject partial-bin requests instead of silently approximating them.
    """
    energy = product["energies"]
    lo, hi = energy["e_low"].to_numpy(), energy["e_high"].to_numpy()
    selected = (lo >= low) & (hi <= high) & np.isfinite(hi)
    if (not selected.any() or lo[selected][0] != low or hi[selected][-1] != high
            or np.any(lo[selected][1:] != hi[selected][:-1])):
        raise ValueError("Band must exactly match contiguous complete energy bins")
    return product["counts_by_energy"][:, selected].sum(axis=1)/product["duration_s"]


def state_summary(product, names, mask=None):
    """Unique native states and changes between sampled rows; no interpolation."""
    rows = []
    mask = np.ones(len(product["time"]), dtype=bool) if mask is None else np.asarray(mask, bool)
    if mask.shape != (len(product["time"]),) or not mask.any():
        raise ValueError("Non-empty sample selection required")
    for name in names:
        values = product["states"][name][mask]
        flattened = values.reshape(len(values), -1)
        changes = np.any(flattened[1:] != flattened[:-1], axis=1)
        unique = np.unique(values, axis=0)
        rows.append(dict(field=name, samples=len(values),
                         unique_states=str(unique.tolist()),
                         changes_between_samples=int(changes.sum())))
    return pd.DataFrame(rows)


def inventory(products):
    """Header provenance and decoded time coverage, deliberately kept separate."""
    return pd.DataFrame([dict(product=p["kind"], file=p["path"].name,
        rows=len(p["time"]), header_start_utc=p["header"]["DATE-BEG"],
        header_end_utc=p["header"]["DATE-END"],
        first_native_utc=(p["start"] if p["kind"] == "cpd" else p["time"])[0].isot,
        last_native_utc=(p["end"] if p["kind"] == "cpd" else p["time"])[-1].isot,
        version=p["header"].get("VERSION"), stixcore=p["header"].get("VERS_SW"),
        spice_kernel=p["header"].get("SPICE_MK"), idb=str(p["idb_versions"])) for p in products])
