import pandas as pd
import os

from functions import (
    convex_function,
    convex_gradient,
    rastrigin_function,
    rastrigin_gradient
)
from experiments import run_experiment

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

    all_results = convex_results + nonconvex_results
    df = pd.DataFrame(all_results)

    df.to_csv("results/summary.csv", index=False)
    print("\nDone. Results saved in results/summary.csv")
    print(df)

if __name__ == "__main__":
    main()