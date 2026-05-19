import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X = iris.data
y = iris.target

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)

labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(10, 7))

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)

colors = plt.cm.get_cmap('rainbow', n_clusters)

for i, label in enumerate(unique_labels):

    if label == -1:
        color = 'black'
        label_name = 'Outliers'

    else:
        color = colors(i)
        label_name = f'Cluster {label}'

    cluster_points = X_scaled[labels == label]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        color=color,
        label=label_name,
        s=100,
        edgecolor='k'
    )

plt.title("DBSCAN Clustering on Iris Dataset (with Cluster Labels)", fontsize=14)

plt.xlabel("Sepal Length (Standardized)", fontsize=12)

plt.ylabel("Sepal Width (Standardized)", fontsize=12)

plt.legend()

plt.show()
