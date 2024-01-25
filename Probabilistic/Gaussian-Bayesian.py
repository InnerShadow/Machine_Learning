import numpy as np
import matplotlib.pyplot as plt

def __main__():
    r1 = 0.8
    var1 = 1.0
    mean1 = [0, -3]
    V1 = [[var1, var1 * r1], [var1 * r1, var1]]

    r2 = 0.7
    var2 = 2.0
    mean2 = [0, 3]
    V2 = [[var2, var2 * r2], [var2 * r2, var2]]

    N = 1000
    x1 = np.random.multivariate_normal(mean1, V1, N).T
    x2 = np.random.multivariate_normal(mean2, V2, N).T

    # empirical estatments
    eMean1 = np.mean(x1.T, axis = 0)
    eMean2 = np.mean(x2.T, axis = 0)

    a = (x1.T - eMean1).T
    eV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                    [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])
    
    a = (x2.T - eMean2).T
    eV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                    [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])
    
    # class apperanse probabilyty and penalty 
    Py1, L1 = 0.5, 1
    Py2, L2 = 1 - Py1, 1

    b = lambda x, v, m, l, py : np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 + np.log(np.linalg.det(v))

    x = np.array([0, -4]) # tp predict
    a = np.argmax([b(x, eV1, eMean1, L1, Py1), b(x, eV2, eMean2, L2, Py2)])
    print(a) 

    plt.figure(figsize = (4, 4))
    plt.title(f"Correlation: r1 = {r1}, r2 = {r2}")
    plt.scatter(x1[0], x1[1], s = 10)
    plt.scatter(x2[0], x2[1], s = 10)
    plt.scatter(x[0], x[1], s = 10)
    plt.show()
   

if __name__ == '__main__':
    __main__()

