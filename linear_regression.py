
import numpy as np

class LinearRegression:
    def init(self):
        self.w = None

    """
    Fits the linear regression model using the normal equation:
    w = (X^T X)^-1 X^T y
    """
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape((-1, 1))
        
        # Add intercept term (bias) to X
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        print("X after adding bias term:")
        print(X_)
        
        # Compute Σ(x_i x_i^T)
        sum_matrix = np.zeros((X_.shape[1], X_.shape[1]))  # Initialize sum as a zero matrix
        for x in X_:
            sum_matrix += np.outer(x, x)  # Use outer product for x_i x_i^T
        
        # Compute X^T X
        XTX = sum_matrix
        XTXInv = np.linalg.pinv(XTX)

        # Compute Σ(y_i x_i) = X^T y
        XTy = X_.T.dot(y)

        # Compute the weights (intercept and slope)
        self.w = XTXInv.dot(XTy)
        return self.w

    def predict(self, X):
        """
        Predicts output for the input data X using the learned weights.
        """
        X = np.array(X)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term (bias)
        return X_.dot(self.w)  # Matrix multiplication to get predictions

# Example Usage
X = np.array([[1,2], [2,3], [3,4], [4,5]])  # Each row is a sample, single feature
y = np.array([3, 5, 7, 9])          # Corresponding target values

model = LinearRegression()
weights = model.fit(X, y)
print("Weights (intercept, slope):", weights.flatten())

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions.flatten())
