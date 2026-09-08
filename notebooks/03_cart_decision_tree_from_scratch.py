import numpy as np
import pandas as pd


class Node:
    """Represents a single node in the Decision Tree structure."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gini=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Predicted class (if leaf node)
        self.gini = gini                # Gini impurity at current node

    def is_leaf(self):
        return self.value is not None


class DecisionTreeCARTFromScratch:
    """Classification and Regression Tree (CART) classifier 
    built from first principles using Gini Impurity.
    """
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        parent_gini = self._gini_impurity(y)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idxs = np.where(X_column <= threshold)[0]
                right_idxs = np.where(X_column > threshold)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate weighted child Gini impurity
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                gini_l = self._gini_impurity(y[left_idxs])
                gini_r = self._gini_impurity(y[right_idxs])
                child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

                # Information Gain in terms of Gini reduction
                gini_reduction = parent_gini - child_gini

                if gini_reduction > best_gain:
                    best_gain = gini_reduction
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check termination criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value, gini=self._gini_impurity(y))

        feature_idxs = list(range(n_features))
        best_feat, best_thresh = self._best_split(X, y, feature_idxs)

        if best_feat is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value, gini=self._gini_impurity(y))

        # Recursive splitting
        left_idxs = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idxs = np.where(X[:, best_feat] > best_thresh)[0]

        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child, gini=self._gini_impurity(y))

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])


if __name__ == "__main__":
    # Load dataset using relative directory path
    df = pd.read_csv('../data/iris.csv')

    # Convert target strings to discrete integer classes
    X = df.iloc[:, :-1].values
    target_names, y = np.unique(df.iloc[:, -1].values, return_inverse=True)

    # Train CART model
    tree = DecisionTreeCARTFromScratch(max_depth=5, min_samples_split=2)
    tree.fit(X, y)

    # Evaluate predictions
    preds = tree.predict(X)
    accuracy = np.mean(preds == y)

    print(f"Target Classes: {list(target_names)}")
    print(f"Root Split Feature Index: {tree.root.feature}")
    print(f"Root Split Threshold:     {tree.root.threshold}")
    print(f"Training Accuracy:        {accuracy * 100:.2f}%")
