import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.unique_classes = np.unique(y)
        
        self.mean = {c: np.mean(X[y == c], axis=0) for c in self.unique_classes}
        self.variance = {c: np.var(X[y == c], axis=0) for c in self.unique_classes}
        self.variance = {c: np.maximum(1e-9, self.variance[c]) for c in self.unique_classes}
        self.prior = {c: np.sum(y == c) / n_samples for c in self.unique_classes}

    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return np.array(preds)

    def _predict(self, x):
        posteriors = [
            self.prior[c] * np.prod(self.probabilityDensityFunc(c, x), axis=0) for c in self.unique_classes
        ]
        return self.unique_classes[np.argmax(posteriors)]

    def probabilityDensityFunc(self, c, x):
        coefficient = 1 / (np.sqrt(2 * np.pi * self.variance[c]))
        return coefficient * np.exp((-(x - self.mean[c]) ** 2) / (2 * self.variance[c]))

    def accuracy_score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == y_test)