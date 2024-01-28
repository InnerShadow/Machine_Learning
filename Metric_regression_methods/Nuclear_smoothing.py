import numpy as np
import matplotlib.pyplot as plt

def Nuclear_smoothing(K, s):
    x = np.arange(0, 10, 0.1)
    x_est = np.arange(0, 10, 0.01)
    N = len(x)
    y_sin = np.sin(x)
    y = y_sin + np.random.normal(0, 0.5, N) # Noize 

    h = 1.0 # window size

    ro = lambda xx, xi: np.abs(xx - xi) # metric
    w = lambda xx, xi: K(ro(xx, xi) / h) # weight

    plt.figure(figsize = (7, 7))
    plot_number = 0

    for h in [0.1, 0.3, 1, 10]:
        y_est = []
        for xx in x_est:
            ww = np.array([w(xx, xi) for xi in x])
            yy = np.dot(ww, y) / sum(ww) # Nadaray-Watson formula
            y_est.append(yy)
        
        plot_number += 1
        plt.subplot(2, 2, plot_number)

        plt.scatter(x, y, color = 'black', s = 10)
        plt.plot(x, y_sin, color = 'blue')
        plt.plot(x_est, y_est, color = 'red')
        plt.title(f'{s} core, h = {h}')
        plt.grid(True)

    plt.show()


def __main__():
    Nuclear_smoothing(lambda r: np.exp(-2 * r * r), 'Gaussian') # Gaussian kernel
    Nuclear_smoothing(lambda r: np.abs(1 - r) * bool(r <= 1), 'Triangle') # Triangle kernel
    Nuclear_smoothing(lambda r: bool(r <= 1), 'Rectangle') # Rectangle kernel
   

if __name__ == '__main__':
    __main__()

