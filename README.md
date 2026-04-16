# Gradient Descent Optimization Project

This project implements gradient descent for both convex and non-convex functions and analyzes how convergence changes with different initial points and learning rates.

## Project Objective

The goal of this project is to study gradient-based optimization by:

- implementing gradient descent from scratch
- testing it on a convex function
- testing it on a non-convex function
- comparing convergence behavior for different starting points
- comparing convergence behavior for different learning rates
- visualizing optimization paths and convergence trends

## Functions Used

### Convex Function

\[
f(x) = x^2 + 2\sin(x)
\]

**Gradient:**

\[
f'(x) = 2x + 2\cos(x)
\]

---

### Non-Convex Function (Rastrigin, 1D)

\[
f(x) = 10 + x^2 - 10\cos(2\pi x)
\]

**Gradient:**

\[
f'(x) = 2x + 20\pi \sin(2\pi x)
\]

## Project Structure

```text
gradient-descent/
│
├── experiments.py
├── functions.py
├── gradient_descent.py
├── main.py
├── plots.py
├── requirements.txt
├── README.md
├── .gitignore
└── results/
```
### Requirements

Install dependencies with:

```pip install -r requirements.txt```

### How to Run

Run the project using:

```python main.py```