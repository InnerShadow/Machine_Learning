import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import matplotlib.pyplot as plt

class KNearestNeighbors:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]

        # get k neighbors
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = np.bincount(k_neighbor_labels).argmax()

        return most_common


def __main__():
    n_neighbors = 7

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    knn_model = KNearestNeighbors(k = n_neighbors)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    print(f'METRICS. My: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = iris.target_names, yticklabels = iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ---------------------------------------------------------------------------------------------------------------------------

    knn_model_sklearn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn_model_sklearn.fit(X_train, y_train)

    y_pred = knn_model_sklearn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')

    print(f'METRICS. SKLEARN: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    cm_sklearn = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm_sklearn, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = iris.target_names, yticklabels = iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Scikit-learn Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    __main__()

