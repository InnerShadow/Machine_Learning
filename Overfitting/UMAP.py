import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array
from sklearn.preprocessing import normalize

def umap(X, n_neighbors = 3, n_components = 2, min_dist = 0.1, n_epochs = 500, learning_rate = 0.5):
    X = check_array(X, dtype = np.float32, accept_sparse = "csr")
    
    # Distance matrix
    distances = pairwise_distances(X, metric = 'euclidean')
    
    # Get KNN for each spot 
    knn_indices = np.argsort(distances, axis = 1)[:, 1 : n_neighbors + 1]
    
    # init with random weights
    random_state = np.random.RandomState(seed = 42)
    init = random_state.uniform(low = -0.5, high = 0.5, size = (X.shape[0], n_components))
    
    for epoch in range(n_epochs):
        # Get weighted sum from each spot
        weighted_sum = np.zeros_like(init)
        for i in range(X.shape[0]):
            for j in knn_indices[i]:
                diff = init[i] - init[j]
                weight = np.exp(-np.sum(diff ** 2) / (2 * min_dist ** 2))
                weighted_sum[i] += weight * diff
        
        # Update weights
        init -= learning_rate * weighted_sum / n_neighbors
        
        # Normolize weights, if need
        # Just remein not normalize freaturs
        
        # init = normalize(init, axis = 1, norm = 'l2')
    
    return init


def __main__():
    s1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], 30)
    s2 = np.random.multivariate_normal([1, 1], [[0.5, 0.3], [0.3, 0.5]], 30)
    s3 = np.random.multivariate_normal([30, 30], [[2, 1], [1, 2]], 30)

    s = np.concatenate([s1, s2, s3])

    plt.scatter(s[:, 0], s[:, 1])
    plt.show()

    um = umap(s)

    plt.scatter(um[:, 0], np.zeros(len(um)))
    plt.show()


if __name__ == '__main__':
    __main__()

