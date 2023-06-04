import numpy as np
import pandas as pd

TRAINING_DATA_PATH = 'elem_stat_learn/data/zip.train'
TEST_DATA_PATH = 'elem_stat_learn/data/zip.test'

CHUNK_SIZE = 10 ** 3


def run():
    X_train, y_train = read_data(TRAINING_DATA_PATH)
    X_test, y_test = read_data(TEST_DATA_PATH)

    X_train_with_intercept = append_intercept(X_train)
    X_test_with_intercept = append_intercept(X_test)

    return X_train_with_intercept, y_train, X_test_with_intercept, y_test


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


def append_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
