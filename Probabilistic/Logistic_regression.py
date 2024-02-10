import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Stochastic gradient descent
        for _ in range(self.num_iterations):
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Do count gradient
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights & bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]

        return predictions


def __mian__():
    N = 50

    X_1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], N)
    y_1 = np.array([0 for _ in range(N)])

    X_2 = np.random.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], N)
    y_2 = np.array([1 for _ in range(N)])

    X = np.concatenate([X_1, X_2])
    y = np.concatenate([y_1, y_2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Predictions:", predictions)

    TP = sum(1 for i, j in zip(predictions, y_test) if i == 1 and j == 1)
    TN = sum(1 for i, j in zip(predictions, y_test) if i == 0 and j == 0)
    FP = sum(1 for i, j in zip(predictions, y_test) if i == 1 and j == 0)
    FN = sum(1 for i, j in zip(predictions, y_test) if i == 0 and j == 1)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Accuracy: {Accuracy}")
    print(f"Recall: {Recall}")
    print(f"Precision: {Precision}")
    print(f"F1: {2 * (Recall * Precision) / (Recall + Precision)}\n")

    # Check using sklearn

    conf_matrix = confusion_matrix(y_test, predictions)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    print(f"Accuracy: {Accuracy}")
    print(f"Recall: {Recall}")
    print(f"Precision: {Precision}")
    print(f"F1: {2 * (Recall * Precision) / (Recall + Precision)}\n")

    recall = recall_score(np.array(y_test), np.array(predictions))
    precision = precision_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}\n")

    plt.scatter(X_test[:, 0], X_test[:, 1], c = predictions - y_test, cmap = 'viridis')
    plt.show()

    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', annot_kws = {"size": 16})

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


if __name__ == "__main__":
    __mian__()

