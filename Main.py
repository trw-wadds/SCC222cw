import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from KNNClassifier import KNNClassifier
from NaiveBayes import NaiveBayes

student_id = 38995239
random.seed(student_id)
header = ["ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]

df = pd.read_csv("glass.data",header=None,names=header)
df = df.drop(df.columns[0], axis=1)
# print(df)

def train_test_split(x, y, test_size=0.2, seed=student_id):
    if len(x) != len(y):
        raise ValueError("There cannot be a different number of features and labels")
    
    data = list(zip(x, y))
    random.shuffle(data)
    x_shuffled, y_shuffled = zip(*data)

    split_point = int(len(x) * (1 - test_size))

    x_train = x_shuffled[:split_point]
    x_test = x_shuffled[split_point:]
    y_train = y_shuffled[:split_point]
    y_test = y_shuffled[split_point:]

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

### train test splot
data = df[header[1:-1]].values
labels = df['Type'].values
x_train, x_test, y_train, y_test = train_test_split(data, labels)

### KNN
# train
knn = KNNClassifier()
train_ts1 = time.time()
knn.fit(x_train, y_train)
train_ts2 = time.time()

# test
test_ts1 = time.time()
predicted_classes = knn.predict(x_test)
test_ts2 = time.time()

# evaluation
accuracy = knn.accuracy_score(x_test, y_test)
print(F"Accuracy: {accuracy * 100:.2f}%")
print(F"Time taken to train: {train_ts2 - train_ts1} seconds"
      F"\nTime taken to test: {test_ts2 - test_ts1} seconds")

### Naïve Bayes
# train
nb = NaiveBayes()
train_ts1 = time.time()
nb.fit(x_train, y_train)
train_ts2 = time.time()

# test
test_ts1 = time.time()
predicted_classes = knn.predict(x_test)
test_ts2 = time.time()

# evaluation
accuracy = nb.accuracy_score(x_test, y_test)
print(F"Accuracy: {accuracy * 100:.2f}%")
print(F"Time taken to train: {train_ts2 - train_ts1} seconds"
      F"\nTime taken to test: {test_ts2 - test_ts1} seconds")

### Decision Tree
# train

train_ts1 = time.time()

train_ts2 = time.time()

# test
test_ts1 = time.time()

test_ts2 = time.time()

# evaluation
accuracy = 0
print(F"Accuracy: {accuracy * 100:.2f}%")
print(F"Time taken to train: {train_ts2 - train_ts1} seconds"
      F"\nTime taken to test: {test_ts2 - test_ts1} seconds")

### Support Vector Machine
# train

train_ts1 = time.time()

train_ts2 = time.time()

# test
test_ts1 = time.time()

test_ts2 = time.time()

# evaluation
accuracy = 0
print(F"Accuracy: {accuracy * 100:.2f}%")
print(F"Time taken to train: {train_ts2 - train_ts1} seconds"
      F"\nTime taken to test: {test_ts2 - test_ts1} seconds")