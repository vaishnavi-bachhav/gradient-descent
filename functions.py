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


# -----------------------------
# Convex function (2D)
# f(x, y) = x^2 + 2 y^2 + x y
# ∇f = (2x + y, 4y + x)
# -----------------------------
def convex2d_function(xy):
    x, y = xy[0], xy[1]
    return x**2 + 2 * y**2 + x * y


def convex2d_gradient(xy):
    x, y = xy[0], xy[1]
    return np.array([2 * x + y, 4 * y + x], dtype=float)


# -----------------------------
# Himmelblau (2D, non-convex)
# f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
# -----------------------------
def himmelblau_function(xy):
    x, y = xy[0], xy[1]
    u = x**2 + y - 11.0
    v = x + y**2 - 7.0
    return u**2 + v**2


def himmelblau_gradient(xy):
    x, y = xy[0], xy[1]
    u = x**2 + y - 11.0
    v = x + y**2 - 7.0
    gx = 4.0 * x * u + 2.0 * v
    gy = 2.0 * u + 4.0 * y * v
    return np.array([gx, gy], dtype=float)