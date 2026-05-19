import pandas as pd
from kmodes.kmodes import KModes

data = {
    'Color': ['Red', 'Blue', 'Green', 'Green', 'Blue', 'Red', 'Red', 'Green'],
    'Shape': ['Circle', 'Square', 'Circle', 'Square', 'Circle', 'Circle', 'Square', 'Square'],
    'Size': ['Small', 'Large', 'Large', 'Small', 'Small', 'Large', 'Small', 'Large']
}

df = pd.DataFrame(data)

print("Original dataset")
print(df)

model = KModes(n_clusters=2, init='Huang', n_init=10, verbose=0)

model.fit(df)

cluster_labels = model.labels_

print("\nCluster labels:")
print(cluster_labels)

print("\nCluster Centroids (Modes)")
print(model.cluster_centroids_)

df['Cluster'] = cluster_labels
print('\n Dataset with cluster')
print(df)
