"""Shared single-epoch observer ephemerides with explicit provenance and caching."""
from datetime import datetime, timezone
import importlib.metadata
import json


def load_observers(epoch, cache_path, refresh=False):
    """Cache a single-epoch Horizons result; never reuse another epoch."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord, frames

    if cache_path.exists() and not refresh:
        stored = json.loads(cache_path.read_text())
        if stored["epoch_utc"] != epoch.utc.isot:
            raise ValueError("Observer cache epoch does not match selected map; use --refresh")
    else:
        targets = {"Solar Orbiter": "-144", "STEREO-A": "-234"}
        coords = {"Earth": get_body_heliographic_stonyhurst("earth", epoch)}
        for name, target in targets.items():
            print(f"Querying JPL Horizons: {name} ({target}) at {epoch.utc.isot} UTC", flush=True)
            coords[name] = get_horizons_coord(target, epoch)
        stored = dict(epoch_utc=epoch.utc.isot, queried_utc=datetime.now(timezone.utc).isoformat(),
                      method="Earth: Astropy built-in through SunPy; spacecraft: JPL Horizons instantaneous heliocentric vectors",
                      endpoint="https://ssd.jpl.nasa.gov/api/horizons_file.api", targets=targets,
                      versions={p: importlib.metadata.version(p) for p in ("sunpy", "astropy")}, observers={})
        for name, coord in coords.items():
            hgs = coord.transform_to(frames.HeliographicStonyhurst(obstime=epoch))
            stored["observers"][name] = dict(lon_deg=hgs.lon.to_value(u.deg), lat_deg=hgs.lat.to_value(u.deg), radius_m=hgs.radius.to_value(u.m))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(stored, indent=2))
    observers = {name: SkyCoord(p["lon_deg"]*u.deg, p["lat_deg"]*u.deg, p["radius_m"]*u.m,
                                frame=frames.HeliographicStonyhurst(obstime=epoch))
                 for name, p in stored["observers"].items()}
    return observers, stored
