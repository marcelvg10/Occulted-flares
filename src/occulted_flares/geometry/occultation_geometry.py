"""Finite-distance occultation threshold above a radially rooted location.

Opaque spherical photosphere; no X-ray opacity model. Handles scalar or array
source locations. This calculates the minimum visible radial height, not an
observed source height. Observer and source must share an epoch and HGS frame.

Derived here from ray/sphere tangency, retaining finite observer distance.
Derivation and relation to the standard distant-observer approximation:
../SOL2026-07-07/Geometry/OCCULTATION_GEOMETRY.md
"""


def radial_occultation(source, observer, opaque_radius):
    import numpy as np
    import astropy.units as u
    from sunpy.coordinates import frames

    observer = observer.transform_to(frames.HeliographicStonyhurst(obstime=source.obstime))
    s = source.cartesian.xyz.to_value(u.km)
    o = observer.cartesian.xyz.to_value(u.km)
    unit_source = s/np.linalg.norm(s, axis=0)
    distance = np.linalg.norm(o)
    radius = opaque_radius.to_value(u.km)
    if not (0 < radius < distance):
        raise ValueError("Observer must lie outside the opaque sphere")
    cos_alpha = np.clip(np.sum(unit_source * o.reshape((3,) + (1,)*(s.ndim-1)), axis=0)/distance, -1, 1)
    alpha = np.arccos(cos_alpha)
    epsilon = radius/distance
    tangent = np.arccos(epsilon)
    hidden = alpha > tangent
    # Tangent-ray intersection with radial line from the assumed flare root:
    # r_crit = R / [sqrt(1-(R/D)^2) sin(alpha) + (R/D) cos(alpha)].
    denominator = np.sqrt(1-epsilon**2)*np.sin(alpha) + epsilon*cos_alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(hidden, np.where(denominator > 0, radius/denominator-radius, np.inf), 0.)
    return dict(center_angle_deg=np.degrees(alpha), theta_occ_deg=np.degrees(alpha)-90.,
                behind_finite_limb_deg=np.degrees(alpha-tangent),
                surface_hidden=hidden, h_occ_km=np.maximum(threshold, 0.))
