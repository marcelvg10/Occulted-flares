"""Source-dependent finite light paths, explicit time conventions and intervals.

Calculations use TDB elapsed seconds, true ICRF/SSB states and straight rays.
UTC is a rendering of the model emission event, not a solar proper-time clock.
Solar rotation, source locations/heights and mission clock validity remain model
assumptions. No stellar/distant-source barycentric correction is applied.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.constants import c

C_KM_S = c.to_value(u.km/u.s)
ROTATION_RAD_S = (14.1844*u.deg/u.day).to_value(u.rad/u.s)


def recover_native(times, convention, applied_shift_s=None):
    """Undo only an explicitly known previously added fixed arrival-time shift."""
    times = Time(times)
    if convention == "native_arrival":
        if applied_shift_s is not None and np.any(np.asarray(applied_shift_s) != 0):
            raise ValueError("Native input cannot also claim a nonzero applied shift")
        return times.copy(), ["Input declared native observer arrival; no prior shift removed"]
    if convention != "earth_equivalent_fixed" or applied_shift_s is None:
        raise ValueError("Unknown timing convention/history: supply the actual previously added shift")
    if not np.isfinite(applied_shift_s).all():
        raise ValueError("Prior shift must be finite")
    native = (times.tai-TimeDelta(applied_shift_s, format="sec", scale="tai")).utc
    return native, [f"Removed explicitly recorded Earth-arrival fixed shift {applied_shift_s} SI seconds"]


def read_stix_intervals(path):
    """Read native L1 CPD bin edges; TIME is a centre offset, not a start."""
    from astropy.io import fits
    path = Path(path)
    with fits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        data_hdu = hdus["DATA"]
        if header.get("LEVEL", "").strip() != "L1":
            raise ValueError("This reader is validated for STIX L1 CPD")
        scale = header.get("TIMESYS", "").strip().lower()
        if scale not in {"utc", "tai", "tt", "tdb"}:
            raise ValueError("Explicit supported TIMESYS required")
        if data_hdu.header.get("TIMEZERO", 0) or any(k in data_hdu.header for k in ("MJDREF", "MJDREFI")):
            raise ValueError("Additional table time reference needs explicit implementation")
        reference = Time(header["DATE-BEG"], scale=scale)
        centre_unit = u.Unit(data_hdu.columns["time"].unit)
        duration_unit = u.Unit(data_hdu.columns["timedel"].unit)
        centre_s = (np.asarray(data_hdu.data["time"], dtype=float)*centre_unit).to_value(u.s)
        duration_s = (np.asarray(data_hdu.data["timedel"], dtype=float)*duration_unit).to_value(u.s)
        if not np.isfinite(centre_s).all() or not np.all(np.isfinite(duration_s) & (duration_s > 0)):
            raise ValueError("Invalid bin centres/durations")
        starts = reference+TimeDelta(centre_s-duration_s/2, format="sec")
        ends = reference+TimeDelta(centre_s+duration_s/2, format="sec")
        record = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      time_scale=scale, time_unit=str(centre_unit), timedel_unit=str(duration_unit),
                      interpretation="DATE-BEG + TIME centre +/- TIMEDEL/2; native arrival",
                      header={k: header.get(k) for k in ("DATE-BEG", "DATE-AVG", "DATE-END", "EAR_TDEL", "SUN_TIME", "DATE_EAR", "DATE_SUN", "XPOSURE", "VERS_SW", "VERSION")},
                      correction_history=["No EAR_TDEL applied to raw DATA by this reader", "Native durations/centres retained; no binning or duration-shift workaround"])
    return dict(start=starts, end=ends, duration_s=duration_s, provenance=record)


def read_euvi_intervals(directory):
    """Use exposure edges from EUVI headers, with UTC interpretation explicit."""
    from astropy.io import fits
    from occulted_flares.geometry.euvi_coordinates import discover
    rows, starts, ends = [], [], []
    for path in discover([Path(directory)]):
        h = fits.getheader(path)
        scale = h.get("TIMESYS", "UTC").strip().lower()
        start = Time(h["DATE-OBS"], scale=scale)
        end = Time(h["DATE-END"], scale=scale) if h.get("DATE-END") else start+float(h["EXPTIME"])*u.s
        if end <= start:
            raise ValueError("Invalid EUVI exposure edges")
        starts.append(start)
        ends.append(end)
        rows.append(dict(file=path.name, wavelength_A=float(h["WAVELNTH"]), exposure_s=float(h["EXPTIME"]),
                         edge_span_s=float((end-start).to_value(u.s)),
                         time_scale=scale, scale_source="TIMESYS" if "TIMESYS" in h else "UTC assumed from SECCHI archive convention",
                         edge_source="DATE-OBS/DATE-END" if h.get("DATE-END") else "DATE-OBS + EXPTIME",
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(start=Time(starts), end=Time(ends), rows=rows)


@dataclass
class SourceModels:
    directions: np.ndarray       # Unit barycentric-axis vectors, shape (3,S)
    anchor_seconds: np.ndarray   # EUVI emission epoch for each spatial hypothesis
    heights_km: np.ndarray
    radius_km: float
    solar_axis: np.ndarray
    sun: object
    rotation_rad_s: float = ROTATION_RAD_S

    @property
    def size(self):
        return self.directions.shape[1]

    def position(self, seconds):
        seconds = np.asarray(seconds, float)
        if seconds.ndim != 2 or seconds.shape[0] != self.size:
            raise ValueError("Source times must have shape (number of models, number of times)")
        angle = self.rotation_rad_s*(seconds-self.anchor_seconds[:, None])
        n = self.directions[:, :, None]
        axis = self.solar_axis[:, None, None]
        perpendicular = np.cross(self.solar_axis, self.directions.T).T[:, :, None]
        parallel = axis*np.sum(axis*n, axis=0)
        rotated = n*np.cos(angle)+perpendicular*np.sin(angle)+parallel*(1-np.cos(angle))
        return self.sun.position(seconds)+rotated*(self.radius_km+self.heights_km)[None, :, None]

    def with_heights(self, heights_km):
        heights = np.broadcast_to(np.asarray(heights_km, float), (self.size,)).copy()
        if np.any(heights < 0):
            raise ValueError("Source heights must be nonnegative")
        return SourceModels(self.directions.copy(), self.anchor_seconds.copy(), heights, self.radius_km,
                            self.solar_axis, self.sun, self.rotation_rad_s)

    def repeat_at_heights(self, heights_km):
        heights = np.asarray(heights_km, float)
        indices = np.repeat(np.arange(self.size), len(heights))
        return SourceModels(self.directions[:, indices], self.anchor_seconds[indices],
                            np.tile(heights, self.size), self.radius_km, self.solar_axis, self.sun, self.rotation_rad_s)


def infer_roots(context_map, ephemerides, tx_arcsec, ty_arcsec, euv_height_km):
    """Intersect contextual sightlines with a retarded Sun-centred sphere.

    WCS/header observer defines LOS orientation. True cached STEREO state is
    used at reception, and Sun position at emission. Near-side intersections
    define the provisional EUV feature/root. This does not calibrate pointing.
    """
    from astropy.coordinates import SkyCoord, ICRS
    from sunpy.coordinates import frames
    tx, ty, heights = np.broadcast_arrays(np.asarray(tx_arcsec, float), np.asarray(ty_arcsec, float), np.asarray(euv_height_km, float))
    tx, ty, heights = tx.ravel(), ty.ravel(), heights.ravel()
    if np.any(heights < 0):
        raise ValueError("EUV heights must be nonnegative")
    radius = context_map.rsun_meters.to_value(u.km)
    epoch = context_map.reference_date
    arrival = float(ephemerides.seconds(epoch))
    directions = SkyCoord(tx*u.arcsec, ty*u.arcsec, np.ones(tx.size)*u.AU, frame=context_map.coordinate_frame).transform_to(ICRS())
    header_observer = context_map.observer_coordinate.transform_to(ICRS()).cartesian.xyz.to_value(u.km)
    los = directions.cartesian.xyz.to_value(u.km)-header_observer[:, None]
    los /= np.linalg.norm(los, axis=0)
    observer = ephemerides.bodies["STEREO-A"].position(arrival)
    emission = np.full(tx.size, arrival-480.)
    for _ in range(20):
        relative = observer[:, None]-ephemerides.bodies["Sun"].position(emission)
        b = np.sum(relative*los, axis=0)
        discriminant = b*b-np.sum(relative*relative, axis=0)+(radius+heights)**2
        if np.any(discriminant < 0):
            raise ValueError("An assumed EUV sightline does not intersect R+h; revise hypotheses")
        distance = -b-np.sqrt(discriminant)
        updated = arrival-distance/C_KM_S
        if np.max(np.abs(updated-emission)) < 1e-9:
            emission = updated
            break
        emission = updated
    else:
        raise ValueError("Retarded EUVI intersection did not converge")
    vectors = observer[:, None]+los*distance-ephemerides.bodies["Sun"].position(emission)
    unit = vectors/np.linalg.norm(vectors, axis=0)
    origin = SkyCoord(x=0*u.km, y=0*u.km, z=0*u.km, representation_type="cartesian", frame=frames.HeliocentricInertial(obstime=epoch)).transform_to(ICRS())
    north = SkyCoord(x=0*u.AU, y=0*u.AU, z=1*u.AU, representation_type="cartesian", frame=frames.HeliocentricInertial(obstime=epoch)).transform_to(ICRS())
    axis = (north.cartesian.xyz-origin.cartesian.xyz).to_value(u.km)
    axis /= np.linalg.norm(axis)
    return SourceModels(unit, emission, heights, radius, axis, ephemerides.bodies["Sun"])


def solve_emission(arrival_seconds, observer, sources, tolerance_s=1e-9):
    """Solve t_emit=t_receive-|observer(t_receive)-source(t_emit)|/c."""
    reception = np.asarray(arrival_seconds, float).ravel()
    receiver = observer.position(reception)[:, None, :]
    initial = np.linalg.norm(receiver[:, 0, :]-sources.sun.position(reception), axis=0)/C_KM_S
    emission = np.broadcast_to(reception-initial, (sources.size, reception.size)).copy()
    for _ in range(20):
        delay = np.linalg.norm(receiver-sources.position(emission), axis=0)/C_KM_S
        updated = reception[None, :]-delay
        if np.max(np.abs(updated-emission)) < tolerance_s:
            return updated
        emission = updated
    raise ValueError("Emission-time fixed point failed to converge")


def solve_arrival(emission_seconds, observer, sources, tolerance_s=1e-9):
    """Predict reception of the same model emission event (not a detection)."""
    emission = np.broadcast_to(np.asarray(emission_seconds).ravel(), (sources.size, np.size(emission_seconds))).copy()
    source_position = sources.position(emission)
    arrival = emission+500.
    for _ in range(20):
        updated = emission+np.linalg.norm(observer.position(arrival)-source_position, axis=0)/C_KM_S
        if np.max(np.abs(updated-arrival)) < tolerance_s:
            return updated
        arrival = updated
    raise ValueError("Arrival-time fixed point failed to converge")


def emission_envelope(arrival_seconds, observer, sources, chunk_size=64):
    """Evaluate all discrete hypotheses at every supplied edge; no time sampling."""
    reception = np.asarray(arrival_seconds).ravel()
    lower, upper = [], []
    for offset in range(0, len(reception), chunk_size):
        emission = solve_emission(reception[offset:offset+chunk_size], observer, sources)
        lower.extend(emission.min(axis=0))
        upper.extend(emission.max(axis=0))
    return np.array(lower), np.array(upper)


def convert_intervals(intervals, ephemerides, observer, nominal, scenarios):
    """Preserve native edges; attach nominal and discrete scenario emission edges."""
    import pandas as pd
    start_s, end_s = ephemerides.seconds(intervals["start"]), ephemerides.seconds(intervals["end"])
    start = solve_emission(start_s, observer, nominal)[0]
    end = solve_emission(end_s, observer, nominal)[0]
    all_edges = np.concatenate([start_s, end_s])
    lo, hi = emission_envelope(all_edges, observer, scenarios)
    n = len(start)
    if not np.all(end > start):
        raise ValueError("Emission interval ordering failed")
    if np.any(start < lo[:n]-1e-6) or np.any(start > hi[:n]+1e-6) or np.any(end < lo[n:]-1e-6) or np.any(end > hi[n:]+1e-6):
        raise ValueError("Nominal model not enclosed by configured source scenarios")
    def utc(seconds):
        t = ephemerides.times(seconds).utc
        t.precision = 6
        return t.isot
    result = pd.DataFrame(dict(native_start_utc=intervals["start"].utc.isot, native_end_utc=intervals["end"].utc.isot,
        emission_start_utc=utc(start), emission_end_utc=utc(end),
        emission_start_tdb_jd=ephemerides.times(start).tdb.jd,
        emission_end_tdb_jd=ephemerides.times(end).tdb.jd,
        emission_start_tdb_jd1=ephemerides.times(start).tdb.jd1,
        emission_start_tdb_jd2=ephemerides.times(start).tdb.jd2,
        emission_end_tdb_jd1=ephemerides.times(end).tdb.jd1,
        emission_end_tdb_jd2=ephemerides.times(end).tdb.jd2,
        native_span_s=end_s-start_s, emission_span_s=end-start,
        start_light_time_s=start_s-start, end_light_time_s=end_s-end,
        emission_start_earliest_utc=utc(lo[:n]), emission_start_latest_utc=utc(hi[:n]),
        emission_end_earliest_utc=utc(lo[n:]), emission_end_latest_utc=utc(hi[n:]),
        start_scenario_spread_s=hi[:n]-lo[:n], end_scenario_spread_s=hi[n:]-lo[n:]))
    return result
