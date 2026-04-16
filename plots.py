import numpy as np
import matplotlib.pyplot as plt
import os

def plot_function_and_path(func, history, title, filename, x_min=-6, x_max=6):
    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = [func(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="Function")
    
    hist_x = np.array(history)
    hist_y = np.array([func(x) for x in hist_x])

    plt.scatter(hist_x, hist_y, marker='o', label="Gradient Descent Steps")
    plt.plot(hist_x, hist_y, linestyle='--')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
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