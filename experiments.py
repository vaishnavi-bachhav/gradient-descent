import numpy as np

from gradient_descent import gradient_descent, gradient_descent_vector
from plots import (
    plot_function_and_path,
    plot_convergence,
    plot_x_vs_iteration,
    plot_2d_contour_path,
    plot_2d_surface_path,
)


def _slug_num(x) -> str:
    """Filesystem-friendly encoding for coordinates and learning rates."""
    return str(x).replace(".", "_").replace("-", "neg")


def _run_id_1d(function_name: str, x0, lr) -> str:
    return f"gd_1d__{function_name}__x0_{_slug_num(x0)}__lr_{_slug_num(lr)}"


def _run_id_2d(function_name: str, x0, y0, lr) -> str:
    return (
        f"gd_2d__{function_name}__x0_{_slug_num(x0)}__y0_{_slug_num(y0)}__lr_{_slug_num(lr)}"
    )


def run_experiment(func, grad_func, function_name, x0_list, lr_list, max_iters=200):
    results = []

    for x0 in x0_list:
        for lr in lr_list:
            final_x, history = gradient_descent(
                func=func,
                grad_func=grad_func,
                x0=x0,
                learning_rate=lr,
                max_iters=max_iters
            )

            final_value = func(final_x)
            num_iters = len(history) - 1

            results.append({
                "function": function_name,
                "x0": x0,
                "learning_rate": lr,
                "final_x": final_x,
                "final_value": final_value,
                "iterations": num_iters
            })

            run_id = _run_id_1d(function_name, x0, lr)

            plot_function_and_path(
                func,
                history,
                title=f"{function_name}: x0={x0}, lr={lr}",
                filename=f"{run_id}__plot__curve_fx_and_descent_path.png",
            )

            plot_convergence(
                history,
                func,
                title=f"Convergence: {function_name}, x0={x0}, lr={lr}",
                filename=f"{run_id}__plot__objective_fx_vs_iteration.png",
            )

            plot_x_vs_iteration(
                history,
                title=f"x vs iteration: {function_name}, x0={x0}, lr={lr}",
                filename=f"{run_id}__plot__state_x_vs_iteration.png",
            )

    return results


def run_experiment_2d(
    func,
    grad_func,
    function_name,
    xy0_list,
    lr_list,
    max_iters=500,
    plot_surface=True,
):
    """
    xy0_list: iterable of (x0, y0) starting points.
    Saves contour (+ optional surface) and objective-vs-iteration plots under results/.
    """
    results = []

    for x0, y0 in xy0_list:
        for lr in lr_list:
            xy0 = np.array([x0, y0], dtype=float)
            final_xy, history = gradient_descent_vector(
                func=func,
                grad_func=grad_func,
                x0=xy0,
                learning_rate=lr,
                max_iters=max_iters,
            )

            final_value = float(func(final_xy))
            num_iters = len(history) - 1

            results.append(
                {
                    "function": function_name,
                    "dim": 2,
                    "x0": float(x0),
                    "y0": float(y0),
                    "learning_rate": lr,
                    "final_x": float(final_xy[0]),
                    "final_y": float(final_xy[1]),
                    "final_value": final_value,
                    "iterations": num_iters,
                }
            )

            run_id = _run_id_2d(function_name, x0, y0, lr)

            plot_2d_contour_path(
                func,
                history,
                title=f"{function_name} (2D): start=({x0}, {y0}), lr={lr}",
                filename=f"{run_id}__plot__contour_fxy_and_descent_path.png",
            )

            if plot_surface:
                plot_2d_surface_path(
                    func,
                    history,
                    title=f"{function_name} (2D surface): start=({x0}, {y0}), lr={lr}",
                    filename=f"{run_id}__plot__surface_fxy_and_descent_path.png",
                )

            plot_convergence(
                history,
                func,
                title=f"Convergence: {function_name} (2D), start=({x0}, {y0}), lr={lr}",
                filename=f"{run_id}__plot__objective_fxy_vs_iteration.png",
            )

    return results