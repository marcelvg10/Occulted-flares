"""Shared SunPy coordinate transformations and finite-distance surface visibility."""


def source_coordinate(row, epoch):
    """Rebuild a saved HGS location with its explicit assumed radius."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    radius = float(row["assumed_radius_km"])*u.km
    return SkyCoord(float(row["hgs_lon_deg"])*u.deg, float(row["hgs_lat_deg"])*u.deg, radius,
                    frame=frames.HeliographicStonyhurst(obstime=epoch, rsun=radius))


def transform_with_check(source, observer):
    """Retain the full 3D point, so even a hidden location round-trips correctly.

    A 2D projected direction alone cannot distinguish near/far surface locations.
    The target distance is retained in the output for that reason.
    """
    import numpy as np
    import astropy.units as u
    from sunpy.coordinates import frames
    target = source.transform_to(frames.Helioprojective(observer=observer,
                                                       obstime=source.obstime,
                                                       rsun=source.radius))
    recovered = target.transform_to(source.frame)
    residual_m = np.linalg.norm((recovered.cartesian.xyz-source.cartesian.xyz).to_value(u.m))
    result = project_surface(source, observer)
    result.update(observer_source_distance_m=float(target.distance.to_value(u.m)),
                  roundtrip_3d_error_m=float(residual_m))
    return result


def project_surface(source, observer):
    """Project a surface point even if occulted; report visibility separately.

    Surface tangent criterion is n dot observer > solar radius, including the
    finite distance to the observer. Positive limb margin means visible surface.
    """
    import numpy as np
    import astropy.units as u
    from sunpy.coordinates import frames

    hpc = source.transform_to(frames.Helioprojective(observer=observer, obstime=source.obstime))
    s = source.cartesian.xyz.to_value(u.m)
    o = observer.cartesian.xyz.to_value(u.m)
    radius = np.linalg.norm(s)
    dot = np.dot(s/radius, o/np.linalg.norm(o))
    separation = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    tangent_angle = np.degrees(np.arccos(radius/np.linalg.norm(o)))
    margin = tangent_angle - separation
    return dict(tx_arcsec=float(hpc.Tx.to_value(u.arcsec)), ty_arcsec=float(hpc.Ty.to_value(u.arcsec)),
                center_angle_deg=float(separation), surface_limb_margin_deg=float(margin),
                surface_visible=bool(margin > 0))
