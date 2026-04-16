from gradient_descent import gradient_descent
from plots import plot_function_and_path, plot_convergence

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

            safe_name = f"{function_name}_x0_{x0}_lr_{lr}".replace(".", "_").replace("-", "neg")
            
            plot_function_and_path(
                func,
                history,
                title=f"{function_name}: x0={x0}, lr={lr}",
                filename=f"{safe_name}_path.png"
            )

            plot_convergence(
                history,
                func,
                title=f"Convergence: {function_name}, x0={x0}, lr={lr}",
                filename=f"{safe_name}_convergence.png"
            )

    return results