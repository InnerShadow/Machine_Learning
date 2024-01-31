import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogramx``
    dendrogram(linkage_matrix, **kwargs)


def __main__():
    x = [(220, 82), (244, 92), (271, 111), (275, 137), (286, 161), 
         (191, 100), (242, 70), (231, 114), (272, 95), (261, 131)]

    x = np.array(x)

    NC = 3 # num of clusters

    clustering = AgglomerativeClustering(n_clusters = NC, linkage = "ward", affinity = 'euclidean')
    x_pr = clustering.fit_predict(x)

    f, ax = plt.subplots(1, 2)
    for c, n in zip(cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), range(NC)):
        clst = x[x_pr == n].T
        ax[0].scatter(clst[0], clst[1], s=10, color=c)

    plot_dendrogram(clustering, ax=ax[1])
    plt.show()
        

if __name__ == '__main__':
    __main__()

