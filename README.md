# Machine Learning from First Principles

A modular implementation of canonical Machine Learning algorithms built from scratch using mathematical ground truths in **NumPy** and **Pandas**.

## Implemented Models

| Model | Optimization / Method | Loss / Objective | Primary Dataset |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Custom Gradient Descent | Binary Cross-Entropy (BCE) | Iris (`data/iris.csv`) |
| **Linear Regression** | Gradient Descent & Analytical Normal Equation | Mean Squared Error (MSE) | Insurance (`data/insurance.csv`) |
| **CART Decision Tree** | Recursive Binary Splitting | Gini Impurity | Iris (`data/iris.csv`) |

## Key Technical Features

* **Zero High-Level ML Frameworks:** Core logic operates purely on NumPy matrix operations without relying on `scikit-learn` wrappers.
* **Numerical Stability Safeguards:** Incorporates floating-point clipping to prevent `np.exp()` overflows and clamping ($\epsilon = 10^{-15}$) for log-loss computations.
* **First-Principles Preprocessing:** Includes custom standardization routines (`CustomStandardScaler`) to prevent feature-scale domination during optimization.

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/Transacttt/ml-from-first-principles.git](https://github.com/Transacttt/ml-from-first-principles.git)
   cd ml-from-first-principles
   python notebooks/01_logistic_regression_from_scratch.py
   python notebooks/02_linear_regression_from_scratch.py
   python notebooks/03_cart_decision_tree_from_scratch.py
