import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from KNNClassifier import KNNClassifier
from NaiveBayes import NaiveBayes
from SVM import SVM

def train_test_split(x, y, seed, test_size=0.2):
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

if __name__ == "__main__":
      student_id = 38995239
      random.seed(student_id)
      header = ["ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]

      df = pd.read_csv("glass.data",header=None,names=header)
      df = df.drop(df.columns[0], axis=1)
      # print(df)


      ### train test split
      data = df[header[1:-1]].values
      labels = df['Type'].values
      x_train, x_test, y_train, y_test = train_test_split(data, labels, student_id)

      # create normalised x values
      x_train_normalised = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + 1e-4) # Normalise X values
      x_test_normalised = (x_test - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + 1e-4) # Normalise X values

      print(len(x_train[0]))
      print(len(x_train_normalised[0]))

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
      print(f"KNN: Accuracy: {accuracy * 100:.2f}%")
      print(f"Time taken to train: {train_ts2 - train_ts1} seconds"
            f"\nTime taken to test: {test_ts2 - test_ts1} seconds")
      
      knn_matrix = confusion_matrix(y_test, predicted_classes)
      disp = ConfusionMatrixDisplay(confusion_matrix=knn_matrix)
      disp.plot()
      plt.title("KNN")

      ### Naïve Bayes
      # train
      nb = NaiveBayes()
      train_ts1 = time.time()
      nb.fit(x_train, y_train)
      train_ts2 = time.time()

      # test
      test_ts1 = time.time()
      predicted_classes = nb.predict(x_test)
      test_ts2 = time.time()

      # evaluation
      accuracy = nb.accuracy_score(x_test, y_test)
      print(f"Naive Bayes: Accuracy: {accuracy * 100:.2f}%")
      print(f"Time taken to train: {train_ts2 - train_ts1} seconds"
            f"\nTime taken to test: {test_ts2 - test_ts1} seconds")
      
      nb_matrix = confusion_matrix(y_test, predicted_classes)
      disp = ConfusionMatrixDisplay(confusion_matrix=nb_matrix)
      disp.plot()
      plt.title("Naive Bayes")

      ### Support Vector Machine
      # train
      svm = SVM()
      train_ts1 = time.time()
      svm.fit(x_train_normalised, y_train)
      train_ts2 = time.time()
      # test
      test_ts1 = time.time()
      predicted_classes = svm.predict(x_test_normalised)
      test_ts2 = time.time()

      # evaluation
      accuracy = svm.accuracy_score(x_test, y_test)
      print(f"SVM: Accuracy: {accuracy * 100:.2f}%")
      print(f"Time taken to train: {train_ts2 - train_ts1} seconds"
            f"\nTime taken to test: {test_ts2 - test_ts1} seconds")
      
      svm_matrix = confusion_matrix(y_test, predicted_classes)
      disp = ConfusionMatrixDisplay(confusion_matrix=svm_matrix)
      disp.plot()
      plt.title("SVM")
      plt.show()

      ### Decision Trees
      # train

      train_ts1 = time.time()

      train_ts2 = time.time()

      # test
      test_ts1 = time.time()

      test_ts2 = time.time()

      # evaluation
      # accuracy = 0
      #print(F"Accuracy: {accuracy * 100:.2f}%")
      #print(F"Time taken to train: {train_ts2 - train_ts1} seconds"
            #F"\nTime taken to test: {test_ts2 - test_ts1} seconds")