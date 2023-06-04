from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

__all__ = ['KNN_ROOT', 'KNN']


def evaluate(labels, pred):
    accuracy = accuracy_score(labels, pred)
    print(f'Accuracy: {accuracy}')


class KNN_ROOT:

    def __init__(self, X_train_with_intercept, y_train, X_test_with_intercept, y_test, k=3):
        self.k = k
        self.X_train = X_train_with_intercept
        self.y_train = y_train
        self.X_test_with_intercept = X_test_with_intercept
        self.y_test = y_test

    def run(self):
        y_pred = self.predict(self.X_test_with_intercept)
        y_pred_train = self.predict(self.X_train)
        print("--- K-Nearest Neighbors\n")
        print("-- Train\n")
        evaluate(self.y_test, y_pred)
        print("-- Test\n")
        evaluate(self.y_train, y_pred_train)

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class KNN:
    def __init__(self, X_train_with_intercept, y_train, X_test_with_intercept, y_test, k):
        self.k = k
        self.X_train = X_train_with_intercept
        self.y_train = y_train
        self.X_test = X_test_with_intercept
        self.y_test = y_test

    def run(self):
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)
        y_pred_train = knn.predict(self.X_train)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print('--- KNN')
        print(f"-- Train\nAccuracy: {test_accuracy}")
        print(f"-- Test\nAccuracy: {train_accuracy}")
        return test_accuracy, train_accuracy


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
