#!/usr/bin/env python3
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from fcmeans import FCM


class Clustering:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x

    def general_clustering(self, cluster, sklearn=True) -> None:
        cluster.fit(self.x)
        if sklearn:
            silhouette_score = metrics.silhouette_score(self.x, cluster.labels_)
        else:
            labels = cluster.predict(self.x)
            silhouette_score = metrics.silhouette_score(self.x, labels)
        print("Silhouette score = ", silhouette_score)

    def k_means(self, n_clusters=8) -> None:
        print("K-means clustering:\n -----------------")
        cluster = KMeans(n_clusters=n_clusters)
        self.general_clustering(cluster)

    def dbscan(self, eps=0.5, min_samples=5) -> None:
        print("DBSCAN clustering:\n -----------------")
        cluster = DBSCAN(eps=eps, min_samples=min_samples)
        self.general_clustering(cluster)

    def fuzzy_c_means(self, n_clusters=8):
        print("fuzzy C-means  clustering:\n -----------------")
        copy_x = self.x
        self.x = self.x.astype(float)
        cluster = FCM(n_clusters=n_clusters)
        self.general_clustering(cluster, sklearn=False)
        self.x = copy_x
