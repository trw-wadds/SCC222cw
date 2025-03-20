import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from KNNClassifier import KNNClassifier

student_id = 38995239
random.seed(student_id)
header = ["ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]

df = pd.read_csv("glass.data",header=None,names=header)
df = df.drop(df.columns[0], axis=1)
# print(df)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop("Type", axis=1)), columns=df.columns[:-1])

# Set up the plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each column and plot its histogram
for i, column in enumerate(df_scaled.columns):
    sns.histplot(df_scaled[column], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {column}')

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()

def train_test_split(x, y, test_size=0.2, seed=student_id):
    if len(x) != len(y):
        raise ValueError("There cannot be a different number of features and labels")
    
    data = list(zip(x, y))
    random.shuffle(data)
    x_shuffled, y_shuffled = zip(*data)

    split_point = int(len(x) * (1 - test_size))

    x_train = x_shuffled[0:split_point]
    x_test = x_shuffled[split_point:len(x)]
    y_train = y_shuffled[0:split_point]
    y_test = y_shuffled[split_point:len(y)]

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
      F"\nTime taken to test: {test_ts2 - train_ts1} seconds")

### Naïve Bayes
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
      F"\nTime taken to test: {test_ts2 - train_ts1} seconds")

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
      F"\nTime taken to test: {test_ts2 - train_ts1} seconds")

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
      F"\nTime taken to test: {test_ts2 - train_ts1} seconds")