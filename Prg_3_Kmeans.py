import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_length','Sepal_width','Petal_length','Petal_width']

y = pd.DataFrame(iris.target)
y.columns = ["Targets"]

model = KMeans(n_clusters=3)
model.fit(X)

plt.figure(figsize=(14,14))
colormap = np.array(['red','lime','black'])

plt.subplot(2,2,1)
plt.scatter(X.Petal_length, X.Petal_width, c=colormap[y.Targets], s=40)
plt.title("Real Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.subplot(2,2,2)
plt.scatter(X.Petal_length, X.Petal_width, c=colormap[model.labels_], s=40)
plt.title("KMeans Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.tight_layout()
plt.show()

print("Cluster centers (centroids):")
print(model.cluster_centers_)
