import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score

TRAINING_DATA_PATH = 'elem_stat_learn/data/zip.train'
TEST_DATA_PATH = 'elem_stat_learn/data/zip.test'

CHUNK_SIZE = 10 ** 3


def read_data(path):
    X, y = [], []
    for chunk in pd.read_csv(path, sep=' ', chunksize=CHUNK_SIZE, header=None):
        y_chunk = chunk.iloc[:, 0]
        X_chunk = chunk.iloc[:, 1:]

        y.append(y_chunk)
        X.append(X_chunk)

    y = pd.concat(y, axis=0)
    X = pd.concat(X, axis=0).dropna(axis=1)
    return X, y


def linear_regression(X_train_with_intercept, X_test_with_intercept, y_train):
    beta = np.linalg.inv(X_train_with_intercept.T.dot(X_train_with_intercept)) \
        .dot(X_train_with_intercept.T) \
        .dot(y_train)

    y_pred_train = X_train_with_intercept.dot(beta)
    y_pred_classes_train = np.vectorize(lambda x: round(x))(y_pred_train)

    y_pred = X_test_with_intercept.dot(beta)
    y_pred_classes = np.vectorize(lambda x: round(x))(y_pred)

    return y_pred, y_pred_classes, y_pred_train, y_pred_classes_train


def evaluate(pred, pred_classes, labels):
    mse = np.mean((pred - labels) ** 2)
    accuracy = accuracy_score(labels, pred_classes)
    recall = recall_score(labels, pred_classes, average='weighted')
    precision = precision_score(labels, pred_classes, average='weighted')
    print(f'MSE: {mse}\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}')


def append_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


if __name__ == '__main__':
    X_train, y_train = read_data(TRAINING_DATA_PATH)
    X_test, y_test = read_data(TEST_DATA_PATH)

    X_train_with_intercept = append_intercept(X_train)
    X_test_with_intercept = append_intercept(X_test)

    y_pred, y_pred_classes, y_pred_train, y_pred_classes_train = linear_regression(X_train_with_intercept,
                                                                                   X_test_with_intercept, y_train)

    print("Train")
    evaluate(y_pred_train, y_pred_classes_train, y_train)
    print("Test")
    evaluate(y_pred, y_pred_classes, y_test)
