import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import os


def _xy_plot_bounds(history, pad_ratio=0.18, min_half_span=2.5):
    """Axis limits around the optimization path (with padding)."""
    h = np.asarray(history, dtype=float)
    if h.ndim == 1:
        h = h.reshape(-1, 2)
    xmin, ymin = h.min(axis=0)
    xmax, ymax = h.max(axis=0)
    span_x = max(float(xmax - xmin), 1e-6)
    span_y = max(float(ymax - ymin), 1e-6)
    px = max(span_x * pad_ratio, min_half_span)
    py = max(span_y * pad_ratio, min_half_span)
    return xmin - px, xmax + px, ymin - py, ymax + py

def plot_function_and_path(func, history, title, filename, x_min=-6, x_max=6):
    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = [func(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(x_vals, y_vals, label="Function", color="C0")

    hist_x = np.asarray(history, dtype=float).ravel()
    hist_y = np.array([func(x) for x in hist_x])

    ax.plot(hist_x, hist_y, linestyle="--", color="C1", alpha=0.85, linewidth=1.0, label="Path")
    ax.scatter(hist_x, hist_y, marker="o", s=28, c="C1", edgecolors="k", linewidths=0.35, zorder=5, label="Steps")

    # Direction arrows along the path in (x, f(x)) space (subsample if very long)
    n = len(hist_x)
    if n >= 2:
        max_arrows = 100
        step = max(1, int(np.ceil((n - 1) / max_arrows)))
        drawn = set()
        for i in range(0, n - 1, step):
            ax.annotate(
                "",
                xy=(hist_x[i + 1], hist_y[i + 1]),
                xytext=(hist_x[i], hist_y[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkred",
                    lw=1.15,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=12,
                ),
                zorder=4,
            )
            drawn.add(i)
        last_i = n - 2
        if last_i not in drawn:
            ax.annotate(
                "",
                xy=(hist_x[last_i + 1], hist_y[last_i + 1]),
                xytext=(hist_x[last_i], hist_y[last_i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkred",
                    lw=1.15,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=12,
                ),
                zorder=4,
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", bbox_inches="tight")
    plt.close()


def plot_x_vs_iteration(history, title, filename):
    """1D parameter trajectory: x vs iteration index (including initial x0 at t=0)."""
    xs = np.asarray(history, dtype=float).ravel()
    it = np.arange(len(xs))

    plt.figure(figsize=(8, 5))
    plt.plot(it, xs, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("x")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", bbox_inches="tight")
    plt.close()


def plot_convergence(history, func, title, filename):
    import os
    import matplotlib.pyplot as plt

    values = [func(x) for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(values)), values, marker='o')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", bbox_inches="tight")
    plt.close()


def plot_2d_contour_path(
    func,
    history,
    title,
    filename,
    levels=50,
    grid_n=180,
    x_lim=None,
    y_lim=None,
):
    """Contour plot of f(x, y) with the gradient-descent path overlaid."""
    hist = np.asarray(history, dtype=float)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 2)

    if x_lim is None or y_lim is None:
        xa0, xa1, ya0, ya1 = _xy_plot_bounds(history)
        x_lim = x_lim or (xa0, xa1)
        y_lim = y_lim or (ya0, ya1)

    xs = np.linspace(x_lim[0], x_lim[1], grid_n)
    ys = np.linspace(y_lim[0], y_lim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts], dtype=float).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    cs = plt.contourf(X, Y, Z, levels=levels, alpha=0.85)
    plt.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.25, alpha=0.35)
    plt.colorbar(cs, fraction=0.046, pad=0.04, label="f(x, y)")

    plt.plot(hist[:, 0], hist[:, 1], "w--", linewidth=1.0, alpha=0.9)
    plt.scatter(hist[:, 0], hist[:, 1], c="red", s=22, edgecolors="k", linewidths=0.3, zorder=5)

    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", bbox_inches="tight")
    plt.close()


def plot_2d_surface_path(
    func,
    history,
    title,
    filename,
    grid_n=55,
    x_lim=None,
    y_lim=None,
):
    """Optional 3D surface z = f(x, y) with the path drawn on the surface."""
    hist = np.asarray(history, dtype=float)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 2)

    if x_lim is None or y_lim is None:
        xa0, xa1, ya0, ya1 = _xy_plot_bounds(history)
        x_lim = x_lim or (xa0, xa1)
        y_lim = y_lim or (ya0, ya1)

    xs = np.linspace(x_lim[0], x_lim[1], grid_n)
    ys = np.linspace(y_lim[0], y_lim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts], dtype=float).reshape(X.shape)

    zp = np.array([func(p) for p in hist], dtype=float)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        rstride=max(1, grid_n // 24),
        cstride=max(1, grid_n // 24),
    )
    ax.plot(hist[:, 0], hist[:, 1], zp, color="red", marker="o", markersize=3, linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", bbox_inches="tight")
    plt.close()