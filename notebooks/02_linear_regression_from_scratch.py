import numpy as np
import pandas as pd


class CustomStandardScaler:
    """Standardizes numeric features by subtracting the mean and dividing by standard deviation."""
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-8
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std


class LinearRegressionFromScratch:
    """Linear Regression model implementing both Gradient Descent and the 
    Analytical Normal Equation solution from mathematical first principles.
    """
    def __init__(self, learning_rate=0.01, epochs=1000, fit_intercept=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.losses = []

    def fit_gradient_descent(self, X, y):
        """Optimizes weights iteratively using Gradient Descent on MSE."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass: y_pred = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute Mean Squared Error (MSE)
            mse = np.mean((y - y_pred) ** 2)
            self.losses.append(mse)

            # Gradients for weights and bias
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def fit_normal_equation(self, X, y):
        """Computes analytical closed-form solution: theta = (X^T X)^(-1) X^T y"""
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X

        # Moore-Penrose pseudo-inverse for numerical stability
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        if self.fit_intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0.0
            self.weights = theta

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def evaluate_regression(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, rmse, r2


if __name__ == "__main__":
    # Load dataset using relative directory path
    df = pd.read_csv('../data/insurance.csv')

    # Encode categorical variables (sex, smoker, region)
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('charges', axis=1).values.astype(float)
    y = df_encoded['charges'].values.astype(float)

    # Standardize numerical features
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Optimization via Gradient Descent
    gd_model = LinearRegressionFromScratch(learning_rate=0.05, epochs=1000)
    gd_model.fit_gradient_descent(X_scaled, y)
    gd_preds = gd_model.predict(X_scaled)
    gd_mse, gd_rmse, gd_r2 = evaluate_regression(y, gd_preds)

    # 2. Closed-form Solution via Normal Equation
    ols_model = LinearRegressionFromScratch()
    ols_model.fit_normal_equation(X_scaled, y)
    ols_preds = ols_model.predict(X_scaled)
    ols_mse, ols_rmse, ols_r2 = evaluate_regression(y, ols_preds)

    print("=== Gradient Descent Results ===")
    print(f"Final RMSE: ${gd_rmse:,.2f}")
    print(f"R^2 Score:  {gd_r2:.4f}\n")

    print("=== Normal Equation (Closed-Form) Results ===")
    print(f"Final RMSE: ${ols_rmse:,.2f}")
    print(f"R^2 Score:  {ols_r2:.4f}")
