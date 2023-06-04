import numpy as np
from sklearn.metrics import accuracy_score

__all__ = ['LINREG']


def evaluate(pred, pred_classes, labels):
    mse = np.mean((pred - labels) ** 2)
    accuracy = accuracy_score(labels, pred_classes)
    print(f'MSE: {mse}\nAccuracy: {accuracy}')


class LINREG:
    def __init__(self, X_train_with_intercept, y_train, X_test_with_intercept, y_test):
        self.X_train_with_intercept = X_train_with_intercept
        self.y_train = y_train
        self.X_test_with_intercept = X_test_with_intercept
        self.y_test = y_test

    def run(self):
        beta = self.fit()
        y_pred, y_pred_classes, y_pred_train, y_pred_classes_train = self.predict(beta)
        print("--- Linear Regression\n")
        print("-- Train\n")
        evaluate(y_pred_train, y_pred_classes_train, self.y_train)
        print("-- Test\n")
        evaluate(y_pred, y_pred_classes, self.y_test)

    def fit(self):
        beta = np.linalg.inv(self.X_train_with_intercept.T.dot(self.X_train_with_intercept)) \
            .dot(self.X_train_with_intercept.T) \
            .dot(self.y_train)

        return beta

    def predict(self, beta):
        y_pred_train = self.X_train_with_intercept.dot(beta)
        y_pred_classes_train = np.vectorize(lambda x: round(x))(y_pred_train)

        y_pred = self.X_test_with_intercept.dot(beta)
        y_pred_classes = np.vectorize(lambda x: round(x))(y_pred)

        return y_pred, y_pred_classes, y_pred_train, y_pred_classes_train
