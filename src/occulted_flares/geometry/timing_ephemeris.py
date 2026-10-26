"""Cached true ICRF/SSB trajectories for photon light-path calculations.

JPL Horizons geometric vectors: spacecraft -144/-234, centre @0, equatorial
ICRF axes, epochs TDB. Sun/Earth use Astropy's explicit built-in ephemeris.
Cubic Hermite interpolation uses velocities and is checked at withheld times.
"""
import json
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
from scipy.interpolate import CubicHermiteSpline


class Trajectory:
    def __init__(self, seconds, positions, velocities):
        self.seconds = np.asarray(seconds, float)
        self.spline = CubicHermiteSpline(self.seconds, np.asarray(positions), np.asarray(velocities), axis=0, extrapolate=False)

    def position(self, seconds):
        seconds = np.asarray(seconds)
        if np.any(seconds < self.seconds[0]) or np.any(seconds > self.seconds[-1]):
            raise ValueError("Time outside cached ephemeris; extrapolation refused")
        return np.moveaxis(self.spline(seconds), -1, 0)


class Ephemerides:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads(self.path.read_text())
        self.reference = Time(self.metadata["reference_utc"], scale="utc").tdb
        self.bodies = {name: Trajectory(record["tdb_seconds"], record["position_km"], record["velocity_km_s"])
                       for name, record in self.metadata["bodies"].items()}

    def seconds(self, times):
        return (Time(times).tdb-self.reference).to_value(u.s)

    def times(self, seconds):
        from astropy.time import TimeDelta
        return self.reference+TimeDelta(seconds, format="sec", scale="tdb")


def download_cache(path, start="2026-07-07T22:15:00", end="2026-07-07T23:20:00", step_s=60):
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    import requests
    import csv
    import io
    import importlib.metadata
    from datetime import datetime, timezone
    path = Path(path)
    if path.exists():
        raise FileExistsError("Existing ephemeris cache preserved; choose another path for refresh")
    reference = Time(start, scale="utc").tdb
    end_time = Time(end, scale="utc").tdb
    seconds = np.arange(0, (end_time-reference).to_value(u.s)+step_s/2, step_s)
    times = reference+seconds*u.s
    # Midpoints withheld from spline construction: independent retrieval/check.
    held_seconds = (seconds[:-1]+seconds[1:])/2
    held_times = reference+held_seconds*u.s
    record = dict(reference_utc=reference.utc.isot, created_utc=datetime.now(timezone.utc).isoformat(),
                  frame="ICRF equatorial axes, solar-system barycentre origin; geometric instantaneous vectors",
                  time_scale="TDB", node_step_seconds=step_s, bodies={}, checks=[],
                  endpoint="https://ssd.jpl.nasa.gov/api/horizons.api",
                  versions={p: importlib.metadata.version(p) for p in ("astropy", "requests", "scipy")})
    for name, target in (("Sun", "sun"), ("Earth", "earth"), ("Solar Orbiter", "-144"), ("STEREO-A", "-234")):
        print("Retrieving geometric trajectory:", name, flush=True)
        responses = []
        def fetch(t):
            if name in {"Sun", "Earth"}:
                with solar_system_ephemeris.set("builtin"):
                    p, v = get_body_barycentric_posvel(target, t)
                return p.xyz.to_value(u.km).T, v.xyz.to_value(u.km/u.s).T
            parameters = dict(format="json", COMMAND=f"'{target}'", EPHEM_TYPE="'VECTORS'", CENTER="'@0'",
                              REF_PLANE="'FRAME'", REF_SYSTEM="'ICRF'", VEC_CORR="'NONE'", VEC_TABLE="'2'",
                              OUT_UNITS="'KM-S'", CSV_FORMAT="'YES'", TIME_TYPE="'TDB'", OBJ_DATA="'NO'",
                              TLIST="'"+"\n".join(f"{jd:.12f}" for jd in t.tdb.jd)+"'")
            response = requests.get(record["endpoint"], params=parameters, timeout=90)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            raw = payload["result"]
            if "$$SOE" not in raw or "$$EOE" not in raw:
                raise ValueError("Horizons response contains no vector table")
            rows = list(csv.reader(io.StringIO(raw.split("$$SOE")[1].split("$$EOE")[0].strip())))
            values = np.array([[float(v) for v in row[2:8]] for row in rows])
            if values.shape != (len(t), 6):
                raise ValueError("Unexpected Horizons vector table shape")
            if not np.allclose([float(row[0]) for row in rows], t.tdb.jd, rtol=0, atol=1e-8):
                raise ValueError("Returned Horizons epochs differ from requested TDB epochs")
            responses.append(dict(parameters=parameters, raw_response=raw))
            p, v = values[:, :3], values[:, 3:6]
            return p, v
        positions, velocities = fetch(times)
        held_positions, _ = fetch(held_times)
        trajectory = Trajectory(seconds, positions, velocities)
        residual_km = np.linalg.norm(trajectory.position(held_seconds).T-held_positions, axis=1)
        # Hold-out error tests interpolation, not absolute orbit/clock accuracy.
        check = dict(body=name, held_out_points=len(held_seconds), max_interpolation_error_km=float(residual_km.max()),
                     equivalent_light_time_error_s=float(residual_km.max()/299792.458))
        if check["equivalent_light_time_error_s"] > 1e-4:
            raise ValueError(f"Trajectory interpolation error exceeds 0.1 ms: {check}")
        record["checks"].append(check)
        record["bodies"][name] = dict(target=target, provider="Astropy builtin (ERFA)" if name in {"Sun", "Earth"} else "JPL Horizons @0/REF_PLANE=FRAME/ICRF/VEC_CORR=NONE",
                                    tdb_seconds=seconds.tolist(), position_km=positions.tolist(), velocity_km_s=velocities.tolist(),
                                    horizons_responses=responses)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print("Saved", path, flush=True)


if __name__ == "__main__":
    import argparse
    from occulted_flares.geometry.geometry_runtime import configure_cache
    configure_cache()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", default="2026-07-07T22:15:00", help="UTC trajectory-window start")
    p.add_argument("--end", default="2026-07-07T23:20:00", help="UTC trajectory-window end")
    p.add_argument("--step", type=float, default=60, help="Node spacing in seconds; midpoint checks are additional")
    args = p.parse_args()
    if args.step <= 0 or Time(args.end,scale="utc") <= Time(args.start,scale="utc"):
        raise ValueError("Require increasing UTC window and positive node spacing")
    download_cache(args.output, args.start, args.end, args.step)
