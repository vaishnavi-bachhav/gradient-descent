import pandas as pd
import os

from functions import (
    convex_function,
    convex_gradient,
    convex2d_function,
    convex2d_gradient,
    rastrigin_function,
    rastrigin_gradient,
    himmelblau_function,
    himmelblau_gradient,
)
from experiments import run_experiment, run_experiment_2d

def main():
    os.makedirs("results", exist_ok=True)

    # Try multiple starting points and learning rates
    x0_list = [-4.0, -2.0, 0.5, 3.0, 5.0]
    lr_list = [0.0001, 0.01, 0.1, 1.0]

    print("Running convex function experiments...")
    convex_results = run_experiment(
        func=convex_function,
        grad_func=convex_gradient,
        function_name="convex",
        x0_list=x0_list,
        lr_list=lr_list,
        max_iters=300
    )

    print("Running Rastrigin function experiments...")
    nonconvex_results = run_experiment(
        func=rastrigin_function,
        grad_func=rastrigin_gradient,
        function_name="rastrigin",
        x0_list=x0_list,
        lr_list=lr_list,
        max_iters=300
    )

    # 2D: several starts; step sizes tuned per landscape
    xy0_list_2d = [
        (-4.0, -4.0),
        (-2.0, 1.0),
        (0.5, -1.5),
        (3.0, 3.0),
        (5.0, -2.0),
    ]
    lr_list_convex2d = [0.01, 0.05, 0.1]
    lr_list_himmelblau = [0.0001, 0.001, 0.005]

    print("Running 2D convex function experiments...")
    convex2d_results = run_experiment_2d(
        func=convex2d_function,
        grad_func=convex2d_gradient,
        function_name="convex2d",
        xy0_list=xy0_list_2d,
        lr_list=lr_list_convex2d,
        max_iters=600,
        plot_surface=True,
    )

    print("Running Himmelblau (2D) experiments...")
    himmelblau_results = run_experiment_2d(
        func=himmelblau_function,
        grad_func=himmelblau_gradient,
        function_name="himmelblau",
        xy0_list=xy0_list_2d,
        lr_list=lr_list_himmelblau,
        max_iters=2500,
        plot_surface=True,
    )

    all_results = (
        convex_results
        + nonconvex_results
        + convex2d_results
        + himmelblau_results
    )
    df = pd.DataFrame(all_results)

    df.to_csv("results/summary.csv", index=False)
    print("\nDone. Results saved in results/summary.csv")
    print(df)

if __name__ == "__main__":
    main()