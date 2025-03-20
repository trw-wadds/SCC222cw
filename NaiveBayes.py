import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.unique_classes = np.unique(y)
        
        self.mean = {c: np.mean(X[y == c], axis=0) for c in self.unique_classes}
        self.variance = {c: np.var(X[y == c], axis=0) for c in self.unique_classes}
        self.prior = {c: np.sum(y == c) / n_samples for c in self.unique_classes}

    def predict(self, X):
        preds = [self.predict(x) for x in X]
        return np.array(preds)

    def _predict(self, X):
        posteriors = []

        posteriors.append(np.prod(self.probabilityDensityFunc(i, X), axis=0) for i in range(len(self.unique_classes)))

    def probabilityDensityFunc(self, class_idx, x):
        pass

    def accuracy_score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == y_test)