import numpy as np
import matplotlib.pyplot as plt

def loss(w, x, y): # sigmoid
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def dfL1(w, x, y): # sigmoid dff for L1
    L1 = 1.0
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + L1 * np.sign(w)


def dfL2(w, x, y): # sigmoid dff for L2
    L1 = 1.0
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + L1 * w


def df(w, x, y): # sigmoid dff
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


def __main__():
    x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
    x_train = [x + [10 * x[0], 10 * x[1], 5 * (x[0] + x[1])] for x in x_train] # more line param
    x_train = np.array(x_train)
    y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

    fn = len(x_train[0])
    n_train = len(x_train)
    wL1 = np.zeros(fn)
    wL2 = np.zeros(fn)
    wSimple = np.zeros(fn)
    nt = 0.00001 # Converge step SGD
    lm = 0.01 # Foggoten step
    N = 5000

    QL1 = np.mean([loss(x, wL1, y) for x, y in zip(x_train, y_train)])
    QL1_plot = [QL1]
    QL2 = np.mean([loss(x, wL2, y) for x, y in zip(x_train, y_train)])
    QL2_plot = [QL2]
    QSimple = np.mean([loss(x, wSimple, y) for x, y in zip(x_train, y_train)])
    QSimple_plot = [QSimple]

    for i in range(N):
        k = np.random.randint(0, n_train - 1)
        ekL1 = loss(wL1, x_train[k], y_train[k])
        wL1 = wL1 - nt * dfL1(wL1, x_train[k], y_train[k])
        ekL2 = loss(wL2, x_train[k], y_train[k])
        wL2 = wL2 - nt * dfL2(wL2, x_train[k], y_train[k])
        ekSimple = loss(wSimple, x_train[k], y_train[k])
        wSimple = wSimple - nt * df(wSimple, x_train[k], y_train[k])
        QL1 = lm * ekL1 + (1 - lm) * QL1
        QL1_plot.append(QL1)
        QL2 = lm * ekL2 + (1 - lm) * QL2
        QL2_plot.append(QL2)
        QSimple = lm * ekSimple + (1 - lm) * QSimple
        QSimple_plot.append(QSimple)

    QL1 = np.mean([loss(x, wL1, y) for x, y in zip(x_train, y_train)])
    print("L1: ")
    print(wL1)
    print(QL1)

    QL2 = np.mean([loss(x, wL2, y) for x, y in zip(x_train, y_train)])
    print("L2: ")
    print(wL2)
    print(QL2)

    QSimple = np.mean([loss(x, wSimple, y) for x, y in zip(x_train, y_train)])
    print("Without regularization: ")
    print(wSimple)
    print(QSimple)

    plt.plot(QL1_plot, color = 'blue', label = 'L1 Regularization')
    plt.plot(QL2_plot, color = 'red', label = 'L2 Regularization')
    plt.plot(QSimple_plot, color = 'green', label = 'Without Regularization')
    
    plt.legend()
    plt.grid(True)
    plt.show()
   

if __name__ == '__main__':
    __main__()

