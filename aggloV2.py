import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# Make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# Create Agglomerative Clustering model
model = AgglomerativeClustering(
    linkage="average", connectivity=connectivity, n_clusters=2
)

# Fit model
model.fit(X)

# Print labels
print(model.labels_)

# Now you can add other features to your data
# For example, let's add a feature that is the sum of the two coordinates
features = [sum(x) for x in X]

# Now you have a separate list of features corresponding to your nodes
# You can refer to them using the labels from your model
for i in range(len(X)):
    print(f"Node {i}, label {model.labels_[i]}, feature {features[i]}")

plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis")
plt.show()
