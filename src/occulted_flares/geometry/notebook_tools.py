"""Notebook presentation and explicit selection storage; science stays in modules.

No plotting function writes files. Inline and widget backends are chosen by the
notebook, never by these helpers. Saving a new accepted selection is explicit.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def show_figure(figure):
    from IPython.display import display
    import matplotlib.pyplot as plt
    display(figure)
    plt.close(figure)


def map_figure(m, points=(), bounds=None):
    """Display original FITS pixels; optionally crop in original pixel units."""
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, AsinhStretch
    from occulted_flares.geometry.geometry_plotting import label_points

    values = m.data
    if bounds is not None:
        x0, x1, y0, y1 = map(int, bounds)
        values = values[max(0, y0):y1+1, max(0, x0):x1+1]
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [1, 99.8])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection=m)
    m.plot(axes=ax, norm=ImageNormalize(vmin=lo, vmax=max(hi, lo+1), stretch=AsinhStretch()))
    if bounds is not None:
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
    else:
        m.draw_limb(axes=ax, color="white")
    points = list(points)
    for row in points:
        ax.plot(float(row["x_pixel"]), float(row["y_pixel"]), "+", color="cyan", ms=8)
    if points:
        label_points(ax, [(float(r["x_pixel"]), float(r["y_pixel"]), r["label"]) for r in points])
    ax.set_title(f"EUVI {m.wavelength} | {m.date.isot}\nOriginal DN; feature identity and calibration unverified", fontsize=10)
    fig.subplots_adjust(bottom=0.16)
    return fig, ax


def measure_points(m, points):
    """Use the same pixel-to-surface calculation as the command-line workflow."""
    from occulted_flares.geometry.euvi_coordinates import measure
    return [measure(m, r["label"], float(r["x_pixel"]), float(r["y_pixel"])) for r in points]


class NotebookPointPicker:
    """Nonblocking widget callback: left click adds; right click removes last.

    Requires %matplotlib widget and a rendered ipympl canvas in VS Code.
    Nothing is saved by clicking. Read .points in a later cell after clicking.
    """
    def __init__(self, m, bounds=None):
        self.figure, self.axes = map_figure(m, bounds=bounds)
        self.points = []
        self._artists = []
        self._connection = self.figure.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        # Do not record clicks while the Matplotlib pan/zoom tool is active.
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if event.button == 1:
            row = dict(label=f"point_{len(self.points)+1}", x_pixel=float(event.xdata), y_pixel=float(event.ydata))
            self.points.append(row)
            marker, = self.axes.plot(row["x_pixel"], row["y_pixel"], "+", color="cyan", ms=9)
            text = self.axes.text(row["x_pixel"]+3, row["y_pixel"]+3, row["label"], color="cyan", fontsize=8)
            self._artists.append((marker, text))
        elif event.button == 3 and self.points:
            self.points.pop()
            for artist in self._artists.pop():
                artist.remove()
        self.figure.canvas.draw_idle()

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self._connection)


def commit_selection(config_path, m, fits_path, points, group_name="A_upper_candidate"):
    """Explicitly accept nonempty surface selections; retain one prior snapshot.

    Also replace configured group membership with these new labels, keeping the
    old group config in selections/previous. Caller must deliberately opt in.
    """
    from occulted_flares.geometry.euvi_coordinates import write_csv
    import matplotlib.pyplot as plt
    import importlib.metadata

    measured = measure_points(m, points)
    labels = [r["label"] for r in measured]
    if not measured or len(set(labels)) != len(labels):
        raise ValueError("Accept a nonempty selection with unique labels")
    if any(r["status"] != "surface_assumed_not_height_measured" for r in measured):
        raise ValueError("Accepted surface geometry needs valid on-disk points")
    config_path, fits_path = Path(config_path).resolve(), Path(fits_path).resolve()
    config = json.loads(config_path.read_text())
    expected = (config_path.parent/config["fits_directory"]/fits_path.name).resolve()
    if fits_path != expected or not fits_path.is_file():
        raise ValueError("Select an original FITS in this event's configured data directory")
    current = config_path.parent/config["selection_run"]
    previous = current.parent/"previous"
    rows = [dict(file=fits_path.name, observation_time=m.date.isot,
                 wcs_reference_time=m.reference_date.isot, **row) for row in measured]
    with tempfile.TemporaryDirectory(prefix="notebook_selection_") as temporary:
        stage = Path(temporary)
        write_csv(stage/"coordinates.csv", rows, list(rows[0]))
        pixels = [dict(file=fits_path.name, **{k: r[k] for k in ("label", "x_pixel", "y_pixel")}) for r in measured]
        write_csv(stage/"selected_pixels.csv", pixels, list(pixels[0]))
        metadata = dict(purpose="Explicitly accepted notebook selections", synthetic_test=False,
                        coordinate_assumption="photosphere; not measured source height",
                        pixel_convention="original FITS zero-based x=column, y=row",
                        versions={p: importlib.metadata.version(p) for p in ("sunpy", "astropy", "matplotlib", "numpy")},
                        files=[dict(path=str(fits_path), sha256=hashlib.sha256(fits_path.read_bytes()).hexdigest(),
                                    date_obs=m.date.isot, wcs_reference_time=m.reference_date.isot)])
        (stage/"metadata.json").write_text(json.dumps(metadata, indent=2))
        fig, _ = map_figure(m, measured)
        fig.savefig(stage/"selected_points.png", dpi=150)
        plt.close(fig)
        previous.mkdir(parents=True, exist_ok=True)
        for name in ("coordinates.csv", "selected_pixels.csv", "metadata.json", "selected_points.png"):
            if (current/name).exists():
                shutil.copy2(current/name, previous/name)
        shutil.copy2(config_path, previous/"selection_groups.json")
        current.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            shutil.copy2(path, current/path.name)
        config["groups"] = {group_name: labels}
        config_path.write_text(json.dumps(config, indent=2))
    return rows
