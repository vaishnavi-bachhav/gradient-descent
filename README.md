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

f(x) = x² + 2 sin(x)

Gradient:
f'(x) = 2x + 2 cos(x)

---

### Non-Convex Function (Rastrigin, 1D)

f(x) = 10 + x² − 10 cos(2πx)

Gradient:
f'(x) = 2x + 20π sin(2πx)

## Project Structure

```text
gradient-descent/
│
├── experiments.py
├── functions.py
├── gradient_descent.py
├── main.py
├── app.py
├── plots.py
├── requirements.txt
├── README.md
├── .gitignore
└── results/
```
### Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### How to run

**Batch experiments (saves figures and `results/summary.csv`):**

```bash
python main.py
```

**Interactive app (Streamlit + Plotly: change objective, learning rate, and starting point in the browser):**

```bash
streamlit run app.py
```

Run that command from the project root (the same folder as `app.py`). Streamlit prints a local URL, usually `http://localhost:8501`, and opens it in your browser if possible. Use the sidebar controls to explore different settings; the plots update when you change a value.