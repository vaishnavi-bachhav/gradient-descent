# Gradient Based Function Optimization

CSI 436/536 Machine Learning – Spring 2026  
**Group 5:**  
Leeshma Adari, Shreya Amagowni, Vaishnavi Bachhav,  Ariana Nieves, Ain e Muhammad  

---

## 📌 Project Overview

This project implements **gradient descent from scratch** and analyzes its behavior on **convex and non-convex functions**.

We study:
- Effect of **learning rate (η)**
- Effect of **initial starting point (x₀)**
- Behavior such as:
  - Convergence
  - Slow learning
  - Oscillation
  - Divergence
  - Local minima trapping

The code reproduces all results shown in the **final report and presentation**.

---

## 📊 Functions Used

### 🔹 Convex (1D)
\[
f(x) = x^2 + 2\sin(x)
\]

Gradient:
\[
f'(x) = 2x + 2\cos(x)
\]

---

### 🔹 Convex (2D)
\[
f(x,y) = x^2 + y^2
\]

Gradient:
\[
\nabla f(x,y) = [2x, 2y]
\]

---

### 🔹 Non-Convex (1D – Rastrigin)
\[
g(x) = 10 + x^2 - 10\cos(2\pi x)
\]

Gradient:
\[
g'(x) = 2x + 20\pi \sin(2\pi x)
\]

---

### 🔹 Non-Convex (2D – Himmelblau)
\[
h(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
\]

---

## ⚙️ Requirements

Python **3.9+ recommended**

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

numpy
matplotlib
pandas
streamlit
plotly

## ▶️ How to Run

### 🔹 Run interactive app 

```bash
streamlit run app.py
```

Open in browser:

http://localhost:8501

You can:

Change function
Change learning rate
Change starting point
Visualize convergence interactively

### 🔹 Run all experiments (optional)

```bash
python main.py
```

This will:

Run gradient descent on all functions
Generate plots
Save outputs in:
``results/``
Save summary file:
``results/summary.csv``



🧪 Experiments Reproduced
✅ Convex 1D

Initial point:

x₀ = 3

Learning rates:

0.0001, 0.01, 0.1, 1.0

Expected behavior:

- 0.0001 → very slow convergence
- 0.01 → smooth convergence
- 0.1 → optimal convergence
- 1.0 → oscillation


✅ Rastrigin (Non-Convex 1D)

Initial points:

x₀ = 3, x₀ = 1.5

Learning rates:

0.0001, 0.01, 0.1

Expected behavior:

- Small η → stuck near local minimum (~2.98)
- Medium η → oscillation
- Results depend on starting point


✅ Convex 2D

Initial point:

(3, 3)

Learning rates:

0.01, 0.2, 0.45

Expected behavior:

- 0.01 → slow
- 0.2 → optimal (reaches (0,0))
- 0.45 → oscillation

✅ Himmelblau (Non-Convex 2D)

Initial point:

(0, 0)

Learning rates:

0.001, 0.005, 0.018

Expected behavior:

- 0.001 → slow
- 0.005 → converges (~130 iterations)
- 0.018 → optimal (~31 iterations, near (3,2))

📈 Output

After running:

results/

Contains:

Convergence plots
Trajectory plots
Contour plots (2D)
CSV summary

⚠️ Notes
Code is written in Python as required by the course.
All results are reproducible from main.py.
Visualization is used to demonstrate:
- convergence behavior
- learning rate impact
- local minima issues