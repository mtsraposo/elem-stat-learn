import matplotlib.pyplot as plt

from elem_stat_learn.src.zipcode_2_8.knn import KNN
from elem_stat_learn.src.zipcode_2_8.linear_regression import LINREG
from elem_stat_learn.src.zipcode_2_8 import data

if __name__ == '__main__':
    train_test_split = data.run()

    lin_reg = LINREG(*train_test_split)
    lin_reg.run()

    neighbors = [1, 3, 5, 7, 15]
    train_accuracies, test_accuracies = [], []
    for k in neighbors:
        knn = KNN(*train_test_split, k=k)
        train_accuracy, test_accuracy = knn.run()
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    fig, ax = plt.subplots()
    ax.plot(neighbors, test_accuracies)
    ax.plot(neighbors, train_accuracies)
    plt.show()



