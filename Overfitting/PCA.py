import numpy as np

def __main__():
    SIZE = 1000
    x = np.random.normal(size = SIZE)
    y = np.random.normal(size = SIZE)
    z = (x + y) / 2 # line dependent feature

    F = np.vstack([x, y, z])
    G = 1 / SIZE * F @ F.T # Gram matrix
    L, W = np.linalg.eig(G)

    sortedW = sorted(zip(L, W.T), key = lambda lx : lx[0], reverse = True)
    sortedW = np.array(w[1] for w in sortedW)

    print(sorted(L, reverse = True))
    

if __name__ == '__main__':
    __main__()

