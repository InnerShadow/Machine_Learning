from sklearn import tree
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def regression():
    max_depth = 3

    x = np.arange(0, np.pi, 0.1).reshape(-1, 1)
    y = np.cos(x)

    clf = tree.DecisionTreeRegressor(max_depth = max_depth)
    clf.fit(x, y)
    y_pred = clf.predict(x)

    plt.plot(x, y, label = 'cos(x)')
    plt.plot(x, y_pred, label = 'DT regression')
    plt.grid(True)
    plt.legend()
    plt.title(f'max deapth = {max_depth}')
    plt.show()

    tree.plot_tree(clf)
    plt.show()


def get_greed(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


def classification():
    max_depth = 5

    iris = datasets.load_iris()
    train_data = np.c_[iris.data[:, 0].reshape(-1, 1), iris.data[:, 2].reshape(-1, 1)]
    train_labels = iris.target

    clf_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth, random_state = 42)
    clf_tree.fit(train_data, train_labels)

    xx, yy = get_greed(train_data)
    ptrdicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, ptrdicted, cmap = 'spring', shading = 'auto')
    plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, s = 50, cmap = 'spring', edgecolors = 'black', linewidths = 1.5)
    plt.show()

    tree.plot_tree(clf_tree)
    plt.show()


def __main__():
    regression()
    classification()


if __name__ == '__main__':
    __main__()

 