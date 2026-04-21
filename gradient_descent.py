import numpy as np


def gradient_descent(func, grad_func, x0, learning_rate, max_iters=200, tol=1e-6):
    x = x0
    history = [x]

    for i in range(max_iters):
        grad = grad_func(x)

        # stop if gradient is very small
        if abs(grad) < tol:
            break

        x = x - learning_rate * grad
        history.append(x)

    return x, history


def gradient_descent_vector(
    func, grad_func, x0, learning_rate, max_iters=200, tol=1e-6
):
    """
    Gradient descent for vector x in R^d (e.g. d=2 for (x, y)).

    x0: array-like of shape (d,)
    grad_func: maps x (ndarray shape (d,)) -> gradient ndarray shape (d,)
    Stops when ||grad|| < tol or max_iters is reached.
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    history = [x.copy()]

    for _ in range(max_iters):
        grad = np.asarray(grad_func(x), dtype=float).reshape(-1)
        if grad.shape != x.shape:
            raise ValueError(
                f"grad shape {grad.shape} does not match x shape {x.shape}"
            )

        if np.linalg.norm(grad) < tol:
            break

        x = x - learning_rate * grad
        history.append(x.copy())

    return x, history