import numpy as np

# -----------------------------
# Convex function
# f(x) = x^2 + 2 sin(x)
# gradient = 2x + 2 cos(x)
# -----------------------------
def convex_function(x):
    return x**2 + 2 * np.sin(x)

def convex_gradient(x):
    return 2 * x + 2 * np.cos(x)


# -----------------------------
# Non-convex function: Rastrigin (1D)
# f(x) = 10 + x^2 - 10 cos(2*pi*x)
# gradient = 2x + 20*pi*sin(2*pi*x)
# -----------------------------
def rastrigin_function(x):
    return 10 + x**2 - 10 * np.cos(2 * np.pi * x)

def rastrigin_gradient(x):
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)