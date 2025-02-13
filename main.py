import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from ste_kmeans import STEClustering


def get_data(n_clusters):
    n_samples = 6000
    n_features = 2
    n_clusters = n_clusters
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state, center_box=(-30.0, 30.0), cluster_std=2)

    print(X.shape, "SH")
    return X


def main():
    n_clusters = 6
    X = get_data(n_clusters )
    random_state = 42

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.subplot(2, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='Set1', s=50, alpha=0.6, edgecolors='k')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

    kmeans = STEClustering(n_clusters)
    y_kmeans = kmeans.fit_predict(X)

    plt.subplot(2, 1, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='Set1', s=50, alpha=0.6, edgecolors='k')
    plt.title("STE Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()


if __name__ == "__main__":
    main()