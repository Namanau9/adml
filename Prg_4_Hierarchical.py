import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()

X = iris.data
y = iris.target

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 7))

dendrogram(Z)

plt.title("Dendogram for Agglomerative Hierarchical Clustering")
plt.ylabel("Distance (Ward's Distance)")
plt.xlabel("Sample")

plt.show()


def divisive_clustering(X, n_clusters=3):

    if n_clusters <= 1:
        return [X]

    model = AgglomerativeClustering(n_clusters=2)

    labels = model.fit_predict(X)

    cluster_1 = X[labels == 0]
    cluster_2 = X[labels == 1]

    result = []

    if len(cluster_1) > 1:
        result += divisive_clustering(cluster_1, n_clusters=n_clusters // 2)

    if len(cluster_2) > 1:
        result += divisive_clustering(cluster_2, n_clusters=n_clusters // 2)

    return result


divisive_result = divisive_clustering(X, n_clusters=3)

plt.figure(figsize=(10, 7))

colors = ['blue', 'green', 'red', 'orange', 'purple']

for idx, cluster in enumerate(divisive_result):

    plt.scatter(
        cluster[:, 0],
        cluster[:, 1],
        label=f'Cluster {idx + 1}',
        color=colors[idx % len(colors)]
    )

plt.title("Divisive Hierarchical Clustering (Top-down Approach)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.legend()

plt.show()
