from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

def __main__():
    n_estimators = 4    
    max_depth = 2

    x = np.arange(0, np.pi, 0.1)
    n_samples = len(x)
    y = np.cos(x) + np.random.normal(0.0, 0.1, n_samples)
    x = x.reshape(-1, 1)

    clf = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    clf.fit(x, y)
    y_pred = clf.predict(x)

    plt.plot(x, y, label = 'cos(x) + noise')
    plt.plot(x, y_pred, label = 'DT Regression')
    plt.grid()
    plt.legend()
    plt.title(f'{n_estimators} trees with {max_depth} depth')
    plt.show()


if __name__ == '__main__':
    __main__()

 