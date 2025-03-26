import numpy as np

class SVM():
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
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
            #print(y)
            i = np.where((y == c1) | (y == c2))[0]
            #print(i)
            # make feature pairings
            feature_pairings_X = X[i]
            feature_pairings_y = y[i]
            #print(feature_pairings_X, feature_pairings_y, "\n\n")
            #print(f"{len(i)}, {len(feature_pairings_X)}, {len(feature_pairings_y)}")
            #print(f"Training hyperplane for classes {c1} vs {c2}")
            hyperPlane = HyperPlane(self.learning_rate, self.lambda_param, self.n_iterations)
            #print(feature_pairings_y)
            hyperPlane.train(feature_pairings_X, feature_pairings_y, n_samples, n_features)
            self.hyperplanes.append(hyperPlane)

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.unique_classes)))
        for h in self.hyperplanes:
            for i, (c1, c2) in enumerate(self.class_pairings):
                preds = h.predict(X)
                votes[:, self.unique_classes.tolist().index(c1)] += (preds == 1) # vote increases if pred is +1
                votes[:, self.unique_classes.tolist().index(c2)] += (preds == -1) # vote decreases if pred is -1
        #print(f"Votes: {votes}")
        return self.unique_classes[np.argmax(votes, axis=1)]

    def accuracy_score(self, x_test, y_test):
        #print(x_test)
        #print(np.ndarray.round(x_test, 1))
        predictions = self.predict(x_test)
        #print(f"Predictions: {predictions}")
        return np.mean(predictions == y_test)
    
class HyperPlane():
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iterations=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.bias = None
        self.weights = None

    def train(self, X, y, n_samples, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0
        unique_labels = np.unique(y)
        #print(y)
        y_decision = np.where(y == unique_labels[0], -1, 1)
        #print(y)
        #print(f"Initial weights: {self.weights}, bias: {self.bias}")

        for i in range(self.n_iterations):
            for id, x in enumerate(X):
                condition = y[id] * (np.dot(x, self.weights) + self.bias)
                #print(y)
                #print(y[id])
                #print(condition)
                if condition >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x, y[id]))
                    self.bias -= self.learning_rate * y[id]
            #if i % 10 == 0:
                #print(f"Iteration {i}: weights: {self.weights}, bias: {self.bias}")

    def predict(self, X):
        #print(f"X: {X}")
        #print(f"Weights: {self.weights}")
        predictions = np.sign(np.dot(X, self.weights) + self.bias)
        #print(f"Predictions: {predictions}")
        return predictions