"""Separate feature-identification evidence from conditional source visibility.

No automatic footpoint identification or source-height inference. Accepted
pixels are treated as hypothetical radial roots only for the geometry test.
"""
import numpy as np
import astropy.units as u
from occulted_flares.geometry.coordinate_transforms import source_coordinate
from occulted_flares.geometry.occultation_geometry import radial_occultation

CLASSES = {"unreviewed", "candidate_low_atmosphere", "candidate_loop", "ambiguous", "artifact_candidate"}


def band_visibility(threshold_Mm, surface_hidden, lower_Mm, upper_Mm):
    """Classify a deterministic height interval, not a probability or detection.

    Grazing a hidden root's threshold is put in the mixed/boundary category.
    A genuinely visible surface root permits zero height as well.
    """
    if not np.isfinite([lower_Mm, upper_Mm]).all() or not 0 <= lower_Mm <= upper_Mm:
        raise ValueError("Trial height band must be finite, ordered and nonnegative")
    if np.isnan(threshold_Mm) or threshold_Mm < 0:
        raise ValueError("Threshold must be nonnegative or infinity")
    if not surface_hidden or lower_Mm > threshold_Mm:
        return "all_trial_heights_clear"
    if upper_Mm < threshold_Mm:
        return "all_trial_heights_hidden"
    return "straddles_or_touches_threshold"


def validate_reviews(labels, reviews):
    if set(reviews)-set(labels):
        raise ValueError("Review refers to unknown selected point labels; check current selection")
    for review in reviews.values():
        classification = review.get("classification", "unreviewed")
        if classification not in CLASSES:
            raise ValueError(f"Unknown feature classification: {classification}")
        if classification != "unreviewed" and (not review.get("evidence", "").strip() or not review.get("alternative", "").strip()):
            raise ValueError("A reviewed feature needs both evidence and an alternative explanation")


def assess_features(m, selected, observers, reviews, height_bands_Mm, high_dn_threshold=16000):
    """Return evidence journal, exact-root thresholds and trial-height outcomes.

    Local DN diagnostics use a 7x7 patch; they are not a calibrated saturation
    mask, a brightness centroid, a flare association or a physical uncertainty.
    """
    validate_reviews([r["label"] for r in selected], reviews)
    if not height_bands_Mm:
        raise ValueError("Provide explicit trial-height bands")
    radius = m.rsun_meters
    epoch = m.reference_date
    journal, thresholds, outcomes = [], [], []
    for row in selected:
        review = reviews.get(row["label"], {})
        classification = review.get("classification", "unreviewed")
        x, y = int(round(float(row["x_pixel"]))), int(round(float(row["y_pixel"])))
        patch = np.asarray(m.data[max(0,y-3):min(m.data.shape[0],y+4), max(0,x-3):min(m.data.shape[1],x+4)])
        finite = patch[np.isfinite(patch)]
        journal.append(dict(label=row["label"], classification=classification,
                            evidence=review.get("evidence", "not yet reviewed"),
                            alternative=review.get("alternative", "not yet reviewed"),
                            footpoint_status="unconfirmed",
                            tx_arcsec=float(row["tx_arcsec"]), ty_arcsec=float(row["ty_arcsec"]),
                            local_DN=float(m.data[y,x]), local_max_DN=float(finite.max()) if finite.size else np.nan,
                            patch_pixels_ge_threshold=int(np.count_nonzero(finite >= high_dn_threshold)),
                            diagnostic_threshold_DN=high_dn_threshold,
                            geometry_caution="Surface-root assumption especially questionable for a loop/artifact" if classification in {"candidate_loop", "artifact_candidate"} else "Hypothetical photospheric root; identity unconfirmed"))
        source = source_coordinate(row, epoch)
        for name, observer in observers.items():
            if abs((observer.obstime-epoch).to_value(u.s)) > 1e-3:
                raise ValueError("Observer and selected-map epochs must match")
            result = radial_occultation(source, observer, radius)
            threshold = float(result["h_occ_km"])/1000
            thresholds.append(dict(label=row["label"], observer=name, classification=classification,
                                   h_occ_Mm=threshold, theta_occ_deg=float(result["theta_occ_deg"]),
                                   surface_hidden=bool(result["surface_hidden"])))
            for band, (lower, upper) in height_bands_Mm.items():
                outcome = band_visibility(threshold, bool(result["surface_hidden"]), lower, upper)
                outcomes.append(dict(label=row["label"], observer=name, classification=classification,
                                     trial_band=band, lower_Mm=lower, upper_Mm=upper,
                                     h_occ_Mm=threshold, outcome=outcome))
    return dict(journal=journal, thresholds=thresholds, outcomes=outcomes)


def plot_assessment(outcomes, height_bands_Mm):
    """Conditional interval matrix, distinct from morphological identification."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    observers = list(dict.fromkeys(r["observer"] for r in outcomes))
    labels = list(dict.fromkeys(r["label"] for r in outcomes))
    bands = list(height_bands_Mm)
    categories = ["all_trial_heights_hidden", "straddles_or_touches_threshold", "all_trial_heights_clear"]
    colors = ["#c45b5b", "#e5b855", "#52a7b8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(1, len(observers), figsize=(7*len(observers), max(5, len(labels)*0.42+2)), squeeze=False)
    for ax, observer in zip(axes[0], observers):
        lookup = {(r["label"], r["trial_band"]): r["outcome"] for r in outcomes if r["observer"] == observer}
        data = [[categories.index(lookup[label, band]) for band in bands] for label in labels]
        ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(len(bands)), [f"{name}\n{height_bands_Mm[name][0]:g}–{height_bands_Mm[name][1]:g} Mm" for name in bands], fontsize=9)
        ax.set_title(observer)
        for i in range(len(labels)):
            for j in range(len(bands)):
                ax.text(j, i, ["hidden", "mixed / tangent", "clear"][data[i][j]], ha="center", va="center", fontsize=9)
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(colors, ["Entire trial band hidden", "Band crosses/touches threshold", "Entire trial band clear"])],
               loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, fontsize=9)
    fig.suptitle("Hypothetical radial-root visibility — feature identities and actual source heights unconfirmed", fontsize=12)
    fig.text(0.5, 0.02, "Illustrative height bands, not measured footpoint/loop populations. Clear does not imply detected; morphology is assessed separately.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.24, wspace=0.3)
    return fig
