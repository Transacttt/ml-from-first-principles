import numpy as np
import pandas as pd


class CustomStandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevent division by zero for zero-variance features
        self.std[self.std == 0] = 1e-8
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std


class LogisticRegressionFromScratch:
    """Logistic Regression classifier built from mathematical ground truths using NumPy."""
    def __init__(self, learning_rate=0.1, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        # Clip z to avoid numerical overflow in np.exp (-500 to 500 range)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize zero weights (valid for convex objective functions)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass: Linear combination -> Sigmoid activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Compute Binary Cross-Entropy (BCE) Loss with epsilon clamping
            eps = 1e-15
            y_pred_clamped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_clamped) + (1 - y) * np.log(1 - y_pred_clamped))
            self.losses.append(loss)

            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    # Load data from relative folder path
    df = pd.read_csv('../data/iris.csv')

    # Prepare binary target (Setosa vs Non-Setosa)
    X = df.iloc[:, :-1].values
    y = (df.iloc[:, -1] == 'Iris-setosa').astype(int).values

    # Preprocess features
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegressionFromScratch(learning_rate=0.1, epochs=300)
    model.fit(X_scaled, y)

    # Evaluate performance
    predictions = model.predict(X_scaled)
    accuracy = np.mean(predictions == y)

    print(f"Initial BCE Loss: {model.losses[0]:.4f}")
    print(f"Final BCE Loss:   {model.losses[-1]:.4f}")
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
