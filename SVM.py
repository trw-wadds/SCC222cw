import numpy

class SVM():
    def __init__():
        pass

    #ASSUME LINEAR SEPERABILITY

    def fit(self, X):
        pass

    def accuracy_score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == y_test)