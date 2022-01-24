#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score


class Clustering:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k_means_clusters: int,
        spectral_clusters: int,
        agglormarative_clusters: int,
    ) -> None:

        """
        Initialize dataset and parameters for Clustering.

        Arguments:
        x: Input features.
        y: Class labels.
        k_means_clusters: Number of clusters for K-means.
        spectral_clusters: Number of clusters for Spectral Clustering.

        Returns:
        None
        """

        self.x = x
        self.y = y
        self.k_means_clusters = k_means_clusters
        self.spectral_clusters = spectral_clusters
        self.agglomerative_clusters = agglormarative_clusters

    def general_clustering(self, cluster) -> np.ndarray:
        """
        Function to perform clustering.

        Arguments:
        cluster: Clustering model.

        Returns:
        cluster.labels_: Cluster labels array.
        """

        cluster.fit(self.x)
        if not cluster.labels_.count(cluster.labels_[0]) == len(cluster.labels_):
            silhouette_score = metrics.silhouette_score(self.x, cluster.labels_)
        else:
            silhouette_score = 0
        mutual_info_score = metrics.normalized_mutual_info_score(
            self.y, cluster.labels_
        )
        print(
            "Silhouette score = ",
            silhouette_score,
            " and normalized mutual info score = ",
            mutual_info_score,
        )
        return cluster.labels_

    def k_means(self, n_clusters=5) -> np.ndarray:
        """
        Function to perform clustering using K-means.

        Returns:
        cluster.labels_: Cluster labels array.
        """

        print("\nK-means clustering:\n -----------------")
        cluster = KMeans(n_clusters=n_clusters, random_state=42)
        return self.general_clustering(cluster)

    def spectral(self, n_clusters=5) -> np.ndarray:
        """
        Function to perform Spectral clustering.

        Returns:
        cluster.labels_: Cluster labels array.
        """

        print("\nSpectral clustering:\n -----------------")
        cluster = SpectralClustering(affinity='nearest_neighbors', n_clusters=n_clusters, random_state=42)
        return self.general_clustering(cluster)

    def agglomerative_clustering(self, linkage="ward", n_clusters=5):
        """
        Function to perform Agglomerative Clustering.

        Arguments:
        n_clusters: Number of clusters.
        linkage: Specify type of linkage.

        Returns:
        cluster.labels_: Cluster labels array.
        """

        print("\nagglomerative clustering:\n -----------------")
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        return self.general_clustering(cluster)

    def optics(self, min_samples=5):
        """
        Function to perform Ordering Points To
        Identify Cluster Structure (OPTICS) clustering.

        Arguments:
        min_samples: Minumum number of samples for clustering.

        Returns:
        cluster.labels_: Cluster labels array.
        """

        print("\nOPTICS clustering:\n -----------------")
        cluster = OPTICS(min_samples=min_samples)
        return self.general_clustering(cluster)

    # def h_dbscan(self, min_samples=5):
    #     cluster = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    #     return self.general_clustering(cluster)

    def cluster_with_plots(self, algorithm="kmeans") -> None:
        """
        Function that plots silhouette scores for range n_clusters
        Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

        Arguments:
        algorithm: the clustering algorithm used

        Returns:
        None
        """
        range_n_clusters = [2, 3, 4, 5, 6]

        for n_clusters in range_n_clusters:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.x) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            if algorithm == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            elif algorithm == "optics":
                clusterer = OPTICS(random_state=10)
            cluster_labels = clusterer.fit_predict(self.x)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self.x, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.show()
