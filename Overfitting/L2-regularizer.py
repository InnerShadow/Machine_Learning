import numpy as np
import matplotlib.pyplot as plt

def L2(L: int, x_values):
    y = np.array([a ** 3 - 10 * a ** 2 + 3 * a + 500 for a in x_values])  # x^3 - 10x^2 + 3x + 500
    x_train, y_train = x_values[::2], y[::2]
    N = 13

    X = np.array([[a ** n for n in range(N)] for a in x_values])
    IL = np.array([[L if i == j else 0 for j in range(N)] for i in range(N)])
    IL[0][0] = 0
    X_train = X[::2]
    Y = y_train

    # w = (XT*X + lambda*I)^-1 * XT * Y
    A = np.linalg.inv(X_train.T @ X_train + IL)
    w = Y @ X_train @ A
    print(w)

    yy = [np.dot(w, x) for x in X]

    plt.plot(x_values, yy, label = f'L = {L}') 


def __main__():
    x_values = np.arange(0, 10.1, 0.1)
    L_values = [5, 10, 20, 25]

    for L in L_values:
        L2(L, x_values)

    y_original = [a ** 3 - 10 * a ** 2 + 3 * a + 500 for a in x_values]
    plt.plot(x_values, y_original, label = 'Original Function')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    __main__()

