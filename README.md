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
16. [**AdaBoostClassification.py**](Boosting/AdaBoostClassification.py) is the binary classification task employing boosting (AdaBoosting). An AdaBoosting algorithm was constructed using number of 'T' decision trees, each with a shallow tree depth (max_depth = 2). With just one decision tree algorithm, there were n = 7 misclassified instances, whereas increasing the ensemble to 3 number of trees resulted in 0 misclassifications. Without ensemble methods, achieving such accurate classification with a limited maximum tree depth would have been challenging. Additionally, a plot was generated where objects with larger weights were represented by points with a larger radius. This is due to the use of an exponential function in AdaBoosting, assigning larger weights to objects that are poorly classified, as visually depicted on the graph.
17. [**AdaBoostRegression.py**](Boosting/AdaBoostRegression.py) is the regression task employing AdaBoostRegression, an ensemble of DecisionTreeRegressor models. In contrast to classification tasks with boosting, a different loss function is used here (exponential for classification and quadratic for regression). Input values for the variable x were generated within the range from zero to π/2, with the target function being the sine function with added noise. Similar to classification, the maximum tree depth remains constant, but the model's generalization capabilities improve through the ensemble of models. Plots were generated for the initially generated target variable, the predicted variable, and the difference between the target and predicted variables. With T = 7 trees, the model provides a sufficiently accurate forecast.
18. [**Word2Vec.ipynb**](NLP/Word2Vec.ipynb) is a natural Language Processing task, word2vec construction. The data consists of relationships between members in the royal family. Auxiliary words were removed from the sentences, and bigrams were formed as all possible pairs of words from the sentences. Next, words were encoded using one-hot encoding and fed into a neural network with 2 neurons in the hidden layer, aiming to establish connections between words. Since 2 neurons in the hidden layer were chosen, the coefficients for each word can be perceived as coordinates on a plane. A plot was generated illustrating how the neural network adjusted the coefficients for each word.
19. [**K-neighbors.py**](Metric_regression_methods/K-neighbors.py) is the classification task using K-nearest neighbors. The model was trained on iris flowers, employing k = 7 nearest neighbors with the Euclidean distance metric. Additionally, the dataset was split into training and testing sets. Post-training, model parameters such as accuracy, precision, recall, and f1-score were evaluated, and a confusion matrix was constructed to visually represent the obtained results. Furthermore, a custom algorithmic model was implemented alongside the utilization of the algorithm from the scikit-learn library.
20. [**UMAP.py**](Overfitting/UMAP.py) task to develop a UMAP algorithm for dimensionality reduction of feature space. In this task, a two-dimensional normal distribution with different parameters was generated, and UMAP attempted to compress the feature space into a simple vector. In each epoch, a weighted sum of differences between a point and its neighbors is calculated. The weights are updated in the direction of this weighted sum.
21. [**tf-idf.py**](NLP/tf-ntf.py) is the task involves finding the tf-idf for the lyrics of Pavel Plamenev in documents p1.txt, p2.txt, p3.txt. Each text underwent lemmatization to obtain more general results, disregarding the number and forms of words. The tf coefficient was calculated as the number of times a word appears in a specific song's lyrics, and the idf coefficient was computed as the logarithm of the ratio of the total number of documents to the number of documents where the word appears.
22. [**Logistic_regression.py**](Probabilistic/Logistic_regression.py) is the logistic regression task for binary classification involved generating two samples from normal bivariate distributions with different means, representing two distinct classes. Subsequently, a logistic regression model with a sigmoid activation function was implemented, utilizing the gradient descent algorithm. Predictions were then made on a test set to determine the class to which each instance belongs. Additionally, metrics such as Accuracy, Recall, Precision, and F1-score were computed. Precision is employed when minimizing false positive predictions is crucial, as in medical applications. Recall is essential to minimize false negatives, especially in antivirus scenarios where missing a virus can have serious consequences. F1-score, a harmonic mean between Recall and Precision, provides a balanced measure of the model's performance. And show why accuracy not the best metric during clases disbalance.
23. [**ROC.py**](Binary_Classification/ROC.py) is the binary classification task using the support vector machine with a linear kernel. After classification, scalar products x * omega were calculated and sorted in reverse order. Subsequently, by increasing number of elements one by one from the sorted array, True Positive Rate (TPR) and False Positive Rate (FPR) were computed, leading to the construction of the ROC curve. Following this, the AUC-ROC and Gini coefficients were computed. It is important to note that this method is not suitable for tasks with class disbalance, as both metrics may be excessively high.
24. [**DecisionTree.py**](/Crucial_trees/DecisionTree.py) is the binary classification task on the dataset of Titanic survivors. The data was loaded from a [*link*](https://www.kaggle.com/datasets/yasserh/titanic-dataset), and uninformative features such as Cabin, Name, PassengerId, Ticket were dropped. Categorical features like Sex and Embarked were numerically encoded using LabelEncoder. Missing values in the age column were replaced with the median, and other missing values were removed. Box plots and histograms were constructed to visualize the distribution, and values of Fare > 100 were discarded as outliers. Subsequently, the dataset was split into training and testing sets, and the values were normalized. A DecisionTree model with a maximum depth of 3 was built, and the rules for decision-making were extracted. Accuracy, Recall, Precision, F1 score, ROC-AUC metrics were calculated, and a confusion matrix was displayed. Additionally, an ROC curve was plotted.