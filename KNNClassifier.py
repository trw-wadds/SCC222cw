import numpy as np

class KNNClassifier():
    def __init__(self, k=6):
        self.k_neighbours = k

    def fit(self, X, y):
        n_samples = np.shape(X)[0]

        if self.k_neighbours > n_samples:
            raise ValueError("K exceeds number of sample points")

        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError("number of features not equal to number of labels")

        # make list of possible classifications
        self.classes = np.unique(y)

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = [self.euclidean_prediction(x) for x in X]
        return np.array(preds)
        

    def euclidean_prediction(self, x):
        distances = [self.euclidean_dist(x, x_train) for x_train in self.X_train]
        # create array of k many labels with shortest distance
        distances = np.c_[distances, self.y_train]
        sorted_distances = distances[distances[:,0].argsort()]
        targets = sorted_distances[0:self.k_neighbours,1]
        
        # return label with highest count
        unique, counts = np.unique(targets, return_counts=True)
        return(unique[np.argmax(counts)])
        

    def euclidean_dist(self, x1, x2):
        return np.sum((x1-x2)**2) # sqrt can be omitted for efficiency as the square root preserves linear ordering (by the intermediate value theorem), which is all we are interested in

    def accuracy_score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == y_test)