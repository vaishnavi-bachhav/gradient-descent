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