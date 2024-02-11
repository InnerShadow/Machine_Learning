import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

def __main__():
    N = 50

    X_1 = np.random.multivariate_normal([2, 1], [[2, 1.5], [1.5, 2]], N)
    y_1 = np.zeros(N)

    X_2 = np.random.multivariate_normal([1, 2], [[2, 1.5], [1.5, 2]], N)
    y_2 = np.ones(N)

    X = np.concatenate([X_1, X_2])
    y = np.concatenate([y_1, y_2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    model = SVC(kernel = 'linear', probability = True)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:\n", conf_matrix)

    # True positive & False positive rates
    TPR = [0]
    FPR = [0]

    thresholds = sorted(probabilities, reverse = True)
    auc = 0
    gini_coefficient = 0

    for threshold in thresholds:
        roc_predictions = (probabilities >= threshold).astype(int)
        TP = np.sum((roc_predictions == 1) & (y_test == 1))
        FP = np.sum((roc_predictions == 1) & (y_test == 0))
        FN = np.sum((roc_predictions == 0) & (y_test == 1))
        TN = np.sum((roc_predictions == 0) & (y_test == 0))

        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))

        auc += TPR[-1] * (FPR[-1] - FPR[-2])

    gini_coefficient = 2 * auc - 1

    print(f"AUC-ROC: {auc}")
    print(f"Gini coefficient: {gini_coefficient}")

    plt.scatter(FPR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    __main__()
