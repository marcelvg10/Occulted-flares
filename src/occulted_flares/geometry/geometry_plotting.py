"""Shared label placement for densely clustered geometry points."""


def label_points(ax, points, color="cyan"):
    """Place labels in a spaced column, with leaders to the unchanged positions.

    points contains (x, y, label) in the axes' data coordinates. Sorting by y
    keeps leaders ordered. No jitter is applied to the scientific coordinates.
    """
    import numpy as np
    points = sorted(points, key=lambda p: p[1])
    for (x, y, label), text_y in zip(points, np.linspace(0.14, 0.86, len(points))):
        ax.annotate(label.replace("point_", ""), xy=(x, y), xycoords="data",
                    xytext=(0.91, text_y), textcoords="axes fraction",
                    color=color, fontsize=10, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white" if color == "black" else "black", ec=color, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
