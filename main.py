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


    # 1D Convex: report uses x0 = 3
    convex_x0_list = [3.0]
    convex_lr_list = [0.0001, 0.01, 0.1, 1.0]

    print("Running convex 1D experiments...")
    convex_results = run_experiment(
        func=convex_function,
        grad_func=convex_gradient,
        function_name="convex",
        x0_list=convex_x0_list,
        lr_list=convex_lr_list,
        max_iters=4000
    )

    # 1D Rastrigin: report uses x0 = 3 and x0 = 1.5
    rastrigin_x0_list = [3.0, 1.5]
    rastrigin_lr_list = [0.0001, 0.01, 0.1]

    print("Running Rastrigin 1D experiments...")
    rastrigin_results = run_experiment(
        func=rastrigin_function,
        grad_func=rastrigin_gradient,
        function_name="rastrigin",
        x0_list=rastrigin_x0_list,
        lr_list=rastrigin_lr_list,
        max_iters=300
    )

    # 2D Convex: report uses start = (3,3)
    convex2d_xy0_list = [(3.0, 3.0)]
    convex2d_lr_list = [0.01, 0.2, 0.45]

    print("Running convex 2D experiments...")
    convex2d_results = run_experiment_2d(
        func=convex2d_function,
        grad_func=convex2d_gradient,
        function_name="convex2d",
        xy0_list=convex2d_xy0_list,
        lr_list=convex2d_lr_list,
        max_iters=4000,
        plot_surface=False,
    )

    # Himmelblau: report uses start = (0,0)
    himmelblau_xy0_list = [(0.0, 0.0)]
    himmelblau_lr_list = [0.001, 0.005, 0.018]

    print("Running Himmelblau 2D experiments...")
    himmelblau_results = run_experiment_2d(
        func=himmelblau_function,
        grad_func=himmelblau_gradient,
        function_name="himmelblau",
        xy0_list=himmelblau_xy0_list,
        lr_list=himmelblau_lr_list,
        max_iters=2500,
        plot_surface=False,
    )

    all_results = (
        convex_results
        + rastrigin_results
        + convex2d_results
        + himmelblau_results
    )

    df = pd.DataFrame(all_results)
    df.to_csv("results/summary.csv", index=False)

    print("\nDone. Results saved in results/summary.csv")
    print(df)


if __name__ == "__main__":
    main()