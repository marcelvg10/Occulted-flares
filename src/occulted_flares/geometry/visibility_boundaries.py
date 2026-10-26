"""Finite-distance visibility boundaries for radially rooted elevated sources.

Opaque photosphere stays at R. For each source height H, construct the circle
of radial roots whose sources at R+H have a ray tangent to that sphere. Project
either the roots or the actual elevated points into the contextual FITS WCS.
No inferred source height, feature association or emission-time alignment.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames


def boundary_coordinates(observer, radius, height, samples=4096):
    """Return (surface roots, elevated tangent sources, critical angle [deg]).

    sin(alpha+beta)=R/(R+H), beta=asin(R/D). Select the far-side
    tangent branch: alpha=pi-asin(R/(R+H))-beta. At H=0 this is acos(R/D).
    """
    if not isinstance(samples, int) or samples < 64:
        raise ValueError("Use at least 64 angular samples")
    r = radius.to_value(u.km)
    h = height.to_value(u.km)
    obs = observer.transform_to(frames.HeliographicStonyhurst(obstime=observer.obstime))
    vector = obs.cartesian.xyz.to_value(u.km)
    distance = np.linalg.norm(vector)
    if not (np.isfinite(h) and h >= 0 and 0 < r <= r+h < distance):
        raise ValueError("Require nonnegative finite source height and observer outside R+H")
    pole = vector/distance
    reference = np.array([0., 0., 1.]) if abs(pole[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(pole, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    alpha = np.pi-np.arcsin(r/(r+h))-np.arcsin(r/distance)
    phase = np.linspace(0, 2*np.pi, samples+1)
    directions = (np.cos(alpha)*pole[:, None]
                  + np.sin(alpha)*(e1[:, None]*np.cos(phase)+e2[:, None]*np.sin(phase)))
    def coordinate(distance_km):
        xyz = directions*distance_km*u.km
        return SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], representation_type="cartesian",
                        frame=frames.HeliographicStonyhurst(obstime=obs.obstime, rsun=radius))
    return coordinate(r), coordinate(r+h), float(np.rad2deg(alpha))


def segment_clearance(points, observer, radius):
    """Independently calculate the closest approach along each finite ray.

    Positive clearance means clear of the opaque sphere. Clip the ray parameter
    to [0,1] so a point in front of the Sun is not blocked by a backward extension.
    """
    obs = observer.transform_to(points.frame)
    p = points.cartesian.xyz.to_value(u.km)
    o = obs.cartesian.xyz.to_value(u.km).reshape(3, *([1]*(p.ndim-1)))
    ray = o-p
    fraction = np.clip(-np.sum(p*ray, axis=0)/np.sum(ray*ray, axis=0), 0, 1)
    closest = p+ray*fraction
    return np.linalg.norm(closest, axis=0)-radius.to_value(u.km)


def calculate_boundaries(m, observers, heights_Mm, samples=4096):
    """Project fresh boundaries into one map at the same WCS epoch."""
    radius = m.rsun_meters
    context_observer = m.observer_coordinate
    curves, checks = [], []
    for name, heights in heights_Mm.items():
        if name not in observers or not heights or len(set(heights)) != len(heights):
            raise ValueError("Need known observers and nonempty unique height lists")
        observer = observers[name]
        if abs((observer.obstime-m.reference_date).to_value(u.s)) > 1e-3:
            raise ValueError("Map WCS and ephemeris epochs must match")
        for height_Mm in heights:
            root, source, angle = boundary_coordinates(observer, radius, height_Mm*u.Mm, samples)
            root_direction = root.transform_to(m.coordinate_frame)
            source_direction = source.transform_to(m.coordinate_frame)
            root_pixels = m.world_to_pixel(root_direction)
            source_pixels = m.world_to_pixel(source_direction)
            # Keep only the near-side photospheric roots visible in context.
            root_xyz = root.cartesian.xyz.to_value(u.km)
            stereo_xyz = context_observer.transform_to(root.frame).cartesian.xyz.to_value(u.km)
            root_visible = np.sum(root_xyz*stereo_xyz[:, None], axis=0) >= radius.to_value(u.km)**2
            # Elevated tangent points can be visible even with hidden roots.
            source_visible = segment_clearance(source, context_observer, radius) >= -1e-5
            error_km = segment_clearance(source, observer, radius)
            curves.append(dict(observer=name, height_Mm=float(height_Mm), root=root, source=source,
                               root_x=root_pixels.x.to_value(u.pix), root_y=root_pixels.y.to_value(u.pix),
                               source_x=source_pixels.x.to_value(u.pix), source_y=source_pixels.y.to_value(u.pix),
                               root_context_visible=root_visible, source_context_visible=source_visible))
            checks.append(dict(observer=name, source_height_Mm=float(height_Mm),
                               opaque_radius_km=radius.to_value(u.km), epoch_utc=m.reference_date.utc.isot,
                               critical_center_angle_deg=angle,
                               max_tangency_error_m=float(np.max(np.abs(error_km)))*1000,
                               root_visible_samples=int(root_visible.sum()), source_visible_samples=int(source_visible.sum())))
    if max(row["max_tangency_error_m"] for row in checks) > 0.01:
        raise ValueError("Boundary ray-tangency check failed (>1 cm)")
    return curves, checks


def plot_boundaries(m, curves, bounds_arcsec=None, mode="root", regions=()):
    """Return a figure and coverage table; never write files or force a backend.

    mode=root: map h_occ=H to surface-root pixels for comparison with root boxes.
    mode=source: project actual tangent-source points at R+H; not root contours.
    Context-hidden curve samples are masked. No missing line is interpolated.
    """
    if mode not in ("root", "source"):
        raise ValueError("mode must be root or source")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from astropy.visualization import ImageNormalize, AsinhStretch

    names = list(dict.fromkeys(c["observer"] for c in curves))
    fig = plt.figure(figsize=(8*len(names), 7.5))
    coverage = []
    palette = ["white", "deepskyblue", "lime", "gold", "magenta", "orange", "violet"]
    for panel, name in enumerate(names, 1):
        ax = fig.add_subplot(1, len(names), panel, projection=m)
        if bounds_arcsec is not None:
            x0, x1, y0, y1 = bounds_arcsec
            corners = SkyCoord([x0, x1, x1, x0]*u.arcsec, [y0, y0, y1, y1]*u.arcsec, frame=m.coordinate_frame)
            pix = m.world_to_pixel(corners)
            xlim = (float(pix.x.value.min()), float(pix.x.value.max()))
            ylim = (float(pix.y.value.min()), float(pix.y.value.max()))
            crop = m.data[max(0, int(ylim[0])):min(m.data.shape[0], int(ylim[1])+1),
                          max(0, int(xlim[0])):min(m.data.shape[1], int(xlim[1])+1)]
        else:
            xlim, ylim = (0, m.data.shape[1]-1), (0, m.data.shape[0]-1)
            crop = m.data
        lo, hi = np.percentile(crop[np.isfinite(crop)], [1, 99.8])
        m.plot(axes=ax, cmap="gray", norm=ImageNormalize(vmin=lo, vmax=max(hi, lo+1), stretch=AsinhStretch()))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        handles = []
        for index, curve in enumerate(c for c in curves if c["observer"] == name):
            color = palette[index % len(palette)]
            style = "-" if curve["height_Mm"] == 0 else "--"
            x, y = curve[f"{mode}_x"], curve[f"{mode}_y"]
            visible = curve[f"{mode}_context_visible"]
            inside = visible & (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
            ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan), color=color,
                    linestyle=style, lw=1.5, path_effects=[effects.withStroke(linewidth=2.4, foreground="black")])
            label = f"{name}: H = {curve['height_Mm']:g} Mm"
            if not np.any(inside):
                label += " (outside view)"
            handles.append(Line2D([], [], color=color, ls=style, label=label))
            coverage.append(dict(observer=name, source_height_Mm=curve["height_Mm"],
                                 projection=mode, boundary_crosses_view=bool(np.any(inside))))
        if mode == "root":
            for region in regions:
                x0, x1, y0, y1 = region["bounds_arcsec"]
                direction = SkyCoord([x0, x1, x1, x0]*u.arcsec, [y0, y0, y1, y1]*u.arcsec, frame=m.coordinate_frame)
                pixels = m.world_to_pixel(direction)
                xy = np.column_stack([pixels.x.value, pixels.y.value])
                ax.add_patch(Polygon(xy, fill=False, edgecolor=region["color"], lw=1))
                ax.text(xy[0, 0], xy[0, 1]-3, region["id"], color=region["color"], fontsize=8)
        ax.legend(handles=handles, loc="lower left", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.85)
        description = "Surface roots where a source at H first clears the Sun" if mode == "root" else "Projected elevated tangent-source positions at R + H"
        ax.set_title(f"{name}\n{description}", fontsize=11)
    fig.suptitle(f"EUVI {m.wavelength} | WCS epoch {m.reference_date.utc.isot} UTC\nFixed opaque photosphere; solid H=0, dashed H>0; heights are hypotheses", fontsize=12)
    fig.text(0.5, 0.025, "Only boundary points visible from the STEREO context viewpoint are shown. No inferred source height, detection claim or photon-time alignment.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.35)
    return fig, coverage
