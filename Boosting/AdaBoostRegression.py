from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

def __main__():
    x = np.arange(0, np.pi / 2, 0.1).reshape(-1, 1)
    y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

    T = 7 # number of algorithms
    max_depth = 2
    algorithms = []
    s = np.array(y.ravel()) # remeins, init like target value

    for n in range(T):
        algorithms.append(DecisionTreeRegressor(max_depth = max_depth))
        algorithms[-1].fit(x, s)

        s -= algorithms[-1].predict(x) # recount remeins

    y_pred = algorithms[0].predict(x)
    for n in range(1, T):
        y_pred += algorithms[n].predict(x)

    plt.plot(x, y, label = 'original data')
    plt.plot(x, y_pred, label = 'predicted data')
    plt.plot(x, s, label = 'diffrence')
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    __main__()

 