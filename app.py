"""
Interactive gradient-descent explorer (Streamlit + Plotly).

Run from the project directory:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from functions import (
    convex_function,
    convex_gradient,
    convex2d_function,
    convex2d_gradient,
    himmelblau_function,
    himmelblau_gradient,
    rastrigin_function,
    rastrigin_gradient,
)
from gradient_descent import gradient_descent, gradient_descent_vector
from plots import _xy_plot_bounds


def _final_grad_norm_1d(grad_func, x):
    return abs(float(grad_func(x)))


def _final_grad_norm_2d(grad_func, xy):
    return float(np.linalg.norm(np.asarray(grad_func(xy), dtype=float)))


def _fig_1d(func, history, x_plot_range: tuple[float, float]):
    xs = np.linspace(x_plot_range[0], x_plot_range[1], 500)
    ys = np.array([func(float(v)) for v in xs], dtype=float)
    hist_x = np.asarray(history, dtype=float).ravel()
    hist_y = np.array([func(float(v)) for v in hist_x], dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="f(x)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=hist_y,
            mode="lines+markers",
            name="Gradient descent",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=7, line=dict(width=0.5, color="white")),
        )
    )
    fig.update_layout(
        title="Objective along x with optimization path",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def _fig_1d_convergence(history, func):
    hist_x = np.asarray(history, dtype=float).ravel()
    fv = np.array([func(float(v)) for v in hist_x], dtype=float)
    it = np.arange(len(fv))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("f(x) vs iteration", "x vs iteration"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=it, y=fv, mode="lines+markers", name="f(x)", marker=dict(size=5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=it, y=hist_x, mode="lines+markers", name="x", marker=dict(size=5)),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=360,
        showlegend=False,
        margin=dict(l=50, r=30, t=70, b=50),
    )
    return fig


def _fig_2d_contour(func, history, pad_ratio=0.12, grid_n=120):
    hist = np.asarray(history, dtype=float)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 2)
    xa0, xa1, ya0, ya1 = _xy_plot_bounds(hist, pad_ratio=pad_ratio, min_half_span=2.0)
    gx = np.linspace(xa0, xa1, grid_n)
    gy = np.linspace(ya0, ya1, grid_n)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts], dtype=float).reshape(X.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=gx,
            y=gy,
            z=Z,
            colorscale="Viridis",
            name="f(x,y)",
            showscale=True,
            colorbar=dict(title="f(x,y)"),
            contours=dict(showlines=True),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist[:, 0],
            y=hist[:, 1],
            mode="lines+markers",
            name="Gradient descent",
            line=dict(color="white", width=2),
            marker=dict(size=8, color="#d62728", line=dict(width=1, color="white")),
        )
    )
    fig.update_layout(
        title="Contour map and descent path",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        height=520,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=40, t=60, b=50),
    )
    return fig


def _fig_2d_surface(func, history, grid_n=52):
    """
    3D view tuned for readability: z-axis is clipped to the path's height range so
    tall Himmelblau "walls" do not squash the trajectory into a flat line.
    """
    hist = np.asarray(history, dtype=float)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 2)
    xa0, xa1, ya0, ya1 = _xy_plot_bounds(hist, pad_ratio=0.12, min_half_span=2.0)
    gx = np.linspace(xa0, xa1, grid_n)
    gy = np.linspace(ya0, ya1, grid_n)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts], dtype=float).reshape(X.shape)
    zp = np.array([func(p) for p in hist], dtype=float)

    z_span = float(np.ptp(zp)) if len(zp) > 1 else max(abs(float(zp[0])), 1.0)
    z_pad = 0.22 * max(z_span, 0.5) + 0.35
    z_lo = float(np.min(zp)) - z_pad
    z_hi = float(np.max(zp)) + z_pad
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            showscale=True,
            opacity=0.38,
            colorbar=dict(title="f(x,y)", thickness=14, len=0.55),
            lighting=dict(
                ambient=0.72,
                diffuse=0.78,
                specular=0.35,
                roughness=0.45,
                fresnel=0.08,
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlight=False,
                    project=dict(z=False),
                    width=1.4,
                    start=z_lo,
                    end=z_hi,
                    size=max((z_hi - z_lo) / 14.0, 1e-4),
                )
            ),
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>f=%{z:.4f}<extra></extra>",
        )
    )

    # Path: thick line + sparse markers so direction is obvious
    n = len(hist)
    step = max(1, n // 35)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    fig.add_trace(
        go.Scatter3d(
            x=hist[:, 0],
            y=hist[:, 1],
            z=zp,
            mode="lines",
            name="Descent path",
            line=dict(color="rgb(255, 65, 40)", width=10),
            hovertemplate="step %{pointNumber}<br>x=%{x:.5f}<br>y=%{y:.5f}<br>f=%{z:.5f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=hist[idx, 0],
            y=hist[idx, 1],
            z=zp[idx],
            mode="markers",
            name="Waypoints",
            marker=dict(
                size=6,
                color="white",
                line=dict(width=1.2, color="rgb(200,40,20)"),
                showscale=False,
            ),
            hovertemplate="x=%{x:.5f}<br>y=%{y:.5f}<br>f=%{z:.5f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[hist[0, 0]],
            y=[hist[0, 1]],
            z=[zp[0]],
            mode="markers+text",
            name="Start",
            marker=dict(size=14, color="#2ca02c", symbol="circle", line=dict(color="white", width=2)),
            text=["start"],
            textposition="top center",
            textfont=dict(size=11, color="#1a1a1a"),
            hovertemplate="<b>Start</b><br>x=%{x:.5f}<br>y=%{y:.5f}<br>f=%{z:.5f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[hist[-1, 0]],
            y=[hist[-1, 1]],
            z=[zp[-1]],
            mode="markers+text",
            name="End",
            marker=dict(size=14, color="#1f77b4", symbol="diamond", line=dict(color="white", width=2)),
            text=["end"],
            textposition="bottom center",
            textfont=dict(size=11, color="#1a1a1a"),
            hovertemplate="<b>End</b><br>x=%{x:.5f}<br>y=%{y:.5f}<br>f=%{z:.5f}<extra></extra>",
        )
    )

    z_axis_title = f"f(x,y)<br><sup>Z clipped to [{z_lo:.3g}, {z_hi:.3g}] along path</sup>"

    fig.update_layout(
        title=dict(
            text="Surface z = f(x, y) with descent path<br>"
            "<sup>Vertical axis is limited to the path height range so the trajectory "
            "is visible (distant peaks are cut off).</sup>",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title=z_axis_title,
            xaxis=dict(backgroundcolor="rgb(245,245,248)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgb(245,245,248)", gridcolor="white", showbackground=True),
            zaxis=dict(
                range=[z_lo, z_hi],
                autorange=False,
                backgroundcolor="rgb(250,250,252)",
                gridcolor="white",
                showbackground=True,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
        ),
        template="plotly_white",
        height=620,
        margin=dict(l=0, r=10, t=95, b=0),
    )
    return fig


def _fig_2d_objective_vs_iter(history, func):
    hist = np.asarray(history, dtype=float)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 2)
    fv = np.array([func(p) for p in hist], dtype=float)
    it = np.arange(len(fv))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=it,
            y=fv,
            mode="lines+markers",
            name="f(x,y)",
            marker=dict(size=5),
        )
    )
    fig.update_layout(
        title="Objective vs iteration",
        xaxis_title="Iteration",
        yaxis_title="f(x, y)",
        template="plotly_white",
        height=320,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


PROBLEMS = {
    "Convex (1D): x² + 2 sin(x)": "convex_1d",
    "Rastrigin (1D)": "rastrigin_1d",
    "Convex (2D): x² + 2y² + xy": "convex_2d",
    "Himmelblau (2D)": "himmelblau_2d",
}


def main():
    st.set_page_config(
        page_title="Gradient Descent Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Gradient descent explorer")
    st.caption(
        "Adjust the objective, starting point, and learning rate. The page reruns and updates the plots."
    )

    with st.sidebar:
        st.header("Controls")
        label = st.selectbox("Objective", list(PROBLEMS.keys()))
        key = PROBLEMS[label]
        is_2d = key.endswith("_2d")

        lr = st.number_input(
            "Learning rate η",
            min_value=1e-6,
            max_value=10.0,
            value=0.1 if not is_2d else 0.005,
            step=1e-6,
            format="%.6f",
            help="Try smaller values on Rastrigin / Himmelblau if the path diverges.",
        )
        max_iters = st.slider("Max iterations", 50, 4000, 600 if is_2d else 400, step=50)
        tol = st.number_input(
            "Stopping tolerance ‖∇f‖",
            min_value=1e-12,
            max_value=1e-1,
            value=1e-6,
            format="%.2e",
        )

        if is_2d:
            x0 = st.slider("Initial x₀", -6.0, 6.0, 3.0, 0.1)
            y0 = st.slider("Initial y₀", -6.0, 6.0, 3.0, 0.1)
        else:
            x0 = st.slider("Initial x₀", -6.0, 6.0, 3.0, 0.1)
            y0 = None

        st.divider()
        st.markdown(
            "**Tip:** On non-convex objectives, compare several η and starting points "
            "to see trapping or divergence."
        )

    # Run optimization
    if key == "convex_1d":
        func, grad = convex_function, convex_gradient
        final_x, history = gradient_descent(
            func, grad, float(x0), lr, max_iters=max_iters, tol=tol
        )
        pad = 1.5
        hist_min = float(np.min(history))
        hist_max = float(np.max(history))
        x_lo = min(-6.5, hist_min - pad)
        x_hi = max(6.5, hist_max + pad)

        c1, c2 = st.columns((1.15, 1.0))
        with c1:
            st.plotly_chart(_fig_1d(func, history, (x_lo, x_hi)), use_container_width=True)
        with c2:
            st.plotly_chart(_fig_1d_convergence(history, func), use_container_width=True)

        gnorm = _final_grad_norm_1d(grad, final_x)
        st.metric("iterations", len(history) - 1)
        mcols = st.columns(4)
        mcols[0].metric("final x", f"{float(final_x):.6f}")
        mcols[1].metric("f(x)", f"{float(func(final_x)):.6f}")
        mcols[2].metric("‖∇f‖ at end", f"{gnorm:.3e}")
        mcols[3].metric(
            "Stopped",
            "tol" if gnorm < tol else ("max iter" if len(history) - 1 >= max_iters else "—"),
        )

    elif key == "rastrigin_1d":
        func, grad = rastrigin_function, rastrigin_gradient
        final_x, history = gradient_descent(
            func, grad, float(x0), lr, max_iters=max_iters, tol=tol
        )
        x_lo, x_hi = -5.5, 5.5
        if len(history) > 0:
            pad = 0.8
            x_lo = min(x_lo, float(np.min(history)) - pad)
            x_hi = max(x_hi, float(np.max(history)) + pad)

        c1, c2 = st.columns((1.15, 1.0))
        with c1:
            st.plotly_chart(_fig_1d(func, history, (x_lo, x_hi)), use_container_width=True)
        with c2:
            st.plotly_chart(_fig_1d_convergence(history, func), use_container_width=True)

        gnorm = _final_grad_norm_1d(grad, final_x)
        st.metric("iterations", len(history) - 1)
        mcols = st.columns(4)
        mcols[0].metric("final x", f"{float(final_x):.6f}")
        mcols[1].metric("f(x)", f"{float(func(final_x)):.6f}")
        mcols[2].metric("‖∇f‖ at end", f"{gnorm:.3e}")
        mcols[3].metric(
            "Stopped",
            "tol" if gnorm < tol else ("max iter" if len(history) - 1 >= max_iters else "—"),
        )

    elif key in ("convex_2d", "himmelblau_2d"):
        if key == "convex_2d":
            func, grad = convex2d_function, convex2d_gradient
        else:
            func, grad = himmelblau_function, himmelblau_gradient

        xy0 = np.array([float(x0), float(y0)], dtype=float)
        final_xy, history = gradient_descent_vector(
            func, grad, xy0, lr, max_iters=max_iters, tol=tol
        )

        st.plotly_chart(_fig_2d_contour(func, history), use_container_width=True)
        st.plotly_chart(_fig_2d_objective_vs_iter(history, func), use_container_width=True)
        with st.expander(
            "3D surface with path (drag to rotate). Z axis is clipped to the path so the curve is visible.",
            expanded=False,
        ):
            st.plotly_chart(_fig_2d_surface(func, history), use_container_width=True)

        gnorm = _final_grad_norm_2d(grad, final_xy)
        st.metric("iterations", len(history) - 1)
        mcols = st.columns(5)
        mcols[0].metric("final x", f"{float(final_xy[0]):.6f}")
        mcols[1].metric("final y", f"{float(final_xy[1]):.6f}")
        mcols[2].metric("f(x,y)", f"{float(func(final_xy)):.6f}")
        mcols[3].metric("‖∇f‖ at end", f"{gnorm:.3e}")
        mcols[4].metric(
            "Stopped",
            "tol" if gnorm < tol else ("max iter" if len(history) - 1 >= max_iters else "—"),
        )


if __name__ == "__main__":
    main()
