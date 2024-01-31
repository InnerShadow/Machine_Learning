# **Machine learning**
1. [**Binary_classification.py**](Binary_Classification/binary_classification.py) is a simple binary classification task using the perceptron algorithm. It features a fixed coefficient omega[1] while updating omega[0]. The primary purpose is to demonstrate basic binary classification with a visually plotted decision boundary at the end.
2. [**FLD.py**](Binary_Classification/FLD.py) is the task involves binary classification without iterative or numerical methods, but rather relies on the mathematical framework, utilizing the Fisher's Linear Discriminant (FLD) loss function.
3. [**SGD.py**](Binary_Classification/SGD.py) is a binary classification task using the gradient descent algorithm and the sigmoid activation function. Stochastic algorithm with pseudo-gradients. Output of the separating hyperplane and a plot of the quality metric (Q) at each iteration at the end of the program.
4. [**L2-regularizer.py**](Overfitting/L2-regularizer.py) is the L2-regularizer task of approximating the original graph with a high-degree polynomial. Enhancing model prediction properties through L2-regularization with varying lambda coefficients, all while preserving the richness of the feature space.
5. [**L1-regularizer.py**](Overfitting/L1-regularizer.py) is the task involves an L1-regularizer with an excessive number of linearly dependent parameters. A comparison of the performance of L1-regularizer, L2-regularizer, and no regularization. The L1-regularizer renders linearly dependent coefficients insignificant.
6. [**Bayesian_inference.py**](Probabilistic/Bayesian_inference.py) is the task involves binary classification using a naive Bayesian approach. It pertains to the probabilistic approach in machine learning tasks and pattern classification.
7. [**Gaussian-Bayesian.py**](Probabilistic/Gaussian-Bayesian.py) is the Gaussian Bayesian classifier for two-dimensional normal random variables. 1000 random variables are generated with parameters m1, m2, var1, var2, and the covariance matrix is calculated based on empirical data. A predictor 'b' is then created, assigning point 'x' to one of the classes.
8. [**SVM.py**](Binary_Classification/SVM.py) is the task concerns the application of the support vector machine (SVM) method in binary classification using the scikit-learn library. Two functions, line_devide and non_line_devide, were created. The former is designed for linearly separable data, while the latter is tailored for linearly non-separable data, with the addition of two outlier values. Support points on both graphs are marked with squares.
9. [**PCA.py**](Overfitting/PCA.py) is the task concerns the principal component analysis (PCA) method. Two values were generated with a normal distribution, and the third was linearly dependent (arithmetic mean of the first two). Subsequently, the Gram matrix was calculated, and eigenvalues and eigenvectors were determined. One of the results of the analysis is: [1.420, 0.992, 0.0].
10. [**Kernel_smoothing.py**](Metric_regression_methods/Kernel_smoothing.py) is the regression task using metric regression methods that lead to the Nadaray-Watson formula. The input data consists of a range of numbers within a specified interval, and the target variable is the sine of x with Gaussian noise. Three kernels were created: Gaussian kernel, Triangle kernel, Rectangle kernel, and four plots were generated with different window sizes for each kernel.
11. [**K-mean.py**](Clustering/K-mean.py) is the сlustering task is performed using the K-means method with Euclidean distance. In this example, when the number of clusters, K, is set to 2, the algorithm makes no mistakes. However, as the number of clusters is increased, even to at least 3, incorrect determination of image membership to clusters and improper selection of cluster center positions begin to occur (as indicated by the large red points on the graph). This happens because the algorithm attempts to cover the feature space with circles (in the case of 2 dimensions).
12. [**DBSCAN.py**](Clustering/DBSCAN.py) is the clustering task using the DBSCAN algorithm. It enables the identification of clusters of arbitrary shapes, such as ribbon-like clusters, and also classifies objects into core, boundary, and noise spots. Also it dismiss the task of determination pre-known number of clusters.
13. [**Hierarchical_clustering.py**](Clustering/Hierarchical_clustering.py) initializes a set of data points in two-dimensional space and performs hierarchical clustering using the AgglomerativeClustering method from scikit-learn. The resulting clusters are visualized in a scatter plot with different colors representing each cluster. Additionally, a dendrogram is generated to illustrate the hierarchical relationships between the clusters.
14. [**CART.py**](Crucial_trees/CART.py) is the classification and regression tasks utilizing crucial trees. For **the regression task**, values along the ordinate axis ranging from 0 to π were generated, corresponding to the target variable values (cosine). Employing the sklearn library, specifically the tree module and the DecisionTreeRegressor model, a decision tree was constructed to predict cosine values at various tree depth levels. Additionally, both the plot of real & predicted values and the decision tree itself were visualized. **The classification task** was implemented using the Iris flower parameters dataset from the sklearn.datasets module. A decision tree for classification was built using the DecisionTreeClassifier from sklearn. A classification plot based on the parameters and the decision tree were generated, with the sklearn.tree.plot_tree() function used to visualize the tree, allowing customization of the maximum tree depth.
15. [**RandomForest.py**](Crucial_trees/RandomForest.py) is the regression task utilizing a random forest (bagging idia) from the sklearn library. An input parameter, x, was generated within the interval [0, π], and the target variable, y, was defined as the cosine of x with added noise. A random forest was constructed using the RandomForestRegressor with varying values for maximum depth and the number of trees. A plot was generated to illustrate the effectiveness of the random forest in predicting values of x.