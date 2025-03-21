import numpy as np

class SVM():
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iterations=100):
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        
        self.hyperplanes = []

    def fit(self, X, y):
        # create list of all unique class pairings
        n_samples, n_features = X.shape
        self.unique_classes = np.unique(y)
        self.class_pairings = [
            (c1, c2)
            for i, c1 in enumerate(self.unique_classes)
            for c2 in self.unique_classes[i+1:]
        ]
        # iterate through, creating a hyperplane for each
        for c1, c2 in self.class_pairings:
            hyperPlane = HyperPlane()
            hyperPlane.train(X, y, n_samples, n_features)
            self.hyperplanes.append(hyperPlane)
        pass

    def predict(self, X):
        for h in self.hyperplanes:
            h._predict(X)

    def accuracy_score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == y_test)
    
class HyperPlane():
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iterations=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.bias = None
        self.weights = None

    def train(self, X, y, n_samples, n_features):
        self.weights = np.zeros(n_samples)
        y_decision = np.where(y <= 0, -1, 1)

        for i in range(self.n_iterations):
            for id, x in enumerate(X):
                if (y_decision[id] * np.dot(x, self.weight) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x, y[id]))
                    self.bias -= self.learning_rate * y[id]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)