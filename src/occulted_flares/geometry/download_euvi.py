"""Search/download STEREO-A EUVI via SunPy Fido/VSO; preserve original FITS.

Query only by default; --download also fetches the matches. Event parameters
live in a separate JSON file. Time bounds are observer arrival UTC, not emission
time. This module does not calibrate data or choose a flare location.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from occulted_flares.geometry.geometry_runtime import configure_cache


def read_config(path):
    """Resolve paths against the event config, independent of working directory."""
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("start_utc", "end_utc", "source", "wavelengths_angstrom", "destination", "reports", "max_files"):
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
    start = datetime.fromisoformat(config["start_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["end_utc"].replace("Z", "+00:00"))
    for timestamp in (start, end):
        if timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("Use UTC timestamps, not local-time offsets")
    if end <= start:
        raise ValueError("end_utc must follow start_utc")
    if config["source"] != "STEREO_A":
        raise ValueError("This initial module is validated for source STEREO_A only")
    waves = config["wavelengths_angstrom"]
    if not waves or any(w not in (171, 195, 284, 304) for w in waves) or len(set(waves)) != len(waves):
        raise ValueError("Use unique EUVI wavelengths: 171, 195, 284, 304")
    if not isinstance(config["max_files"], int) or config["max_files"] < 1:
        raise ValueError("max_files must be a positive integer")
    for key in ("destination", "reports"):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def inspect_download(path):
    """Validate FITS readability and record headers without altering raw data."""
    from astropy.io import fits
    record = dict(path=str(path.resolve()), bytes=path.stat().st_size)
    with path.open("rb") as stream:
        record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    with fits.open(path, checksum=True) as hdus:
        hdus.verify("exception")
        header = hdus[0].header
        record["header"] = {key: str(header.get(key, "")) for key in (
            "INSTRUME", "DETECTOR", "OBSRVTRY", "TELESCOP", "WAVELNTH", "WAVEUNIT", "DATE-OBS", "DATE-END", "EXPTIME", "LEVEL")}
        record["has_image"] = any(hdu.data is not None and hdu.data.ndim == 2 for hdu in hdus)
        if not record["has_image"]:
            raise ValueError("Downloaded FITS contains no 2D image")
    return record


def run(config_path, download):
    config = read_config(config_path)
    configure_cache()
    import astropy.units as u
    from astropy.table import QTable
    from sunpy.net import Fido, attrs as a

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_dir = Path(config["reports"]) / "latest"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(created_utc=stamp, config_path=str(config_path.resolve()), config=config,
                    mode="download" if download else "search_only", status="started",
                    versions={name: importlib.metadata.version(name) for name in ("sunpy", "astropy", "parfive")},
                    queries=[], files=[], errors=[])
    manifest_path = report_dir / "manifest.json"
    try:
        searches = []
        for wave in config["wavelengths_angstrom"]:
            print(f"Searching STEREO-A EUVI {wave} Angstrom, {config['start_utc']} to {config['end_utc']} UTC...", flush=True)
            response = Fido.search(a.Time(config["start_utc"], config["end_utc"]),
                                   a.Instrument("EUVI"), a.Source(config["source"]),
                                   a.Wavelength(wave*u.angstrom))
            count = sum(len(table) for table in response)
            manifest["queries"].append(dict(wavelength_angstrom=wave, count=count))
            print(response, flush=True)
            for index, table in enumerate(response):
                # VSO tables carry a live client in metadata; preserve columns,
                # not that unserializable Python service object.
                saved_table = QTable(table)
                saved_table.meta = {}
                saved_table.write(report_dir / f"search_{wave}_{index}.ecsv", format="ascii.ecsv", overwrite=True)
            searches.append(response)
        count = sum(q["count"] for q in manifest["queries"])
        manifest["match_count"] = count
        if not count:
            manifest["status"] = "no_matches"
            print("No VSO matches. This does not prove no observations exist; archive indexing/availability needs checking.")
            return 2
        if count > config["max_files"]:
            raise ValueError(f"{count} matches exceed max_files={config['max_files']}; narrow the config or raise the explicit limit")
        if not download:
            manifest["status"] = "search_complete"
            print("Search saved; add --download to fetch these matching observations.")
            return 0
        destination = Path(config["destination"])
        destination.mkdir(parents=True, exist_ok=True)
        seen = set()
        for response in searches:
            if not sum(len(table) for table in response):
                continue
            results = Fido.fetch(response, path=str(destination / "{file}"), overwrite=False, max_conn=2, progress=True)
            manifest["errors"].extend(str(error) for error in results.errors)
            for name in results:
                path = Path(name).resolve()
                if path in seen:
                    continue
                seen.add(path)
                try:
                    manifest["files"].append(inspect_download(path))
                except Exception as error:
                    manifest["errors"].append(f"Invalid file {path}: {error}")
        manifest["status"] = "partial_failure" if manifest["errors"] else "download_complete"
        print(f"Verified {len(manifest['files'])} files in {destination}. Errors: {len(manifest['errors'])}.")
        return 1 if manifest["errors"] else 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(error).__name__}: {error}")
        print(f"Search/download failed: {error}", flush=True)
        return 1
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Report: {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.config, args.download)
    except (ValueError, OSError, ImportError) as error:
        print(f"Cannot start: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
