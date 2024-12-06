import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import cm

wine_df = pd.read_csv('wine_no_label.csv')
X = wine_df.values

sc = StandardScaler()
X_std = sc.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

feature_names = [f"Principal Component {i+1}" for i in range(2)]

kmeans = KMeans(n_clusters=3, random_state=0)
y_km = kmeans.fit_predict(X_pca)

dbscan = DBSCAN(eps=0.55, min_samples=5, metric='euclidean')
y_db = dbscan.fit_predict(X_pca)

agg_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
y_agg = agg_cluster.fit_predict(X_pca)

def plot_clusters(X, y, title):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Cluster 1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Cluster 2')
    if len(np.unique(y)) > 2:
        plt.scatter(X[y == 2, 0], X[y == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='Cluster 3')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_dbscan(X, y, title):
    unique_labels = np.unique(y)
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        if label == -1:
            col = 'k'
            label_name = 'Noise'
        else:
            col = plt.cm.Spectral(float(label) / len(unique_labels))
            label_name = f'Cluster {label + 1}'
        plt.scatter(X[y == label, 0], X[y == label, 1], s=50, c=col, marker='o', edgecolor='black', label=label_name)
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_dendrogram(X, title):
    row_clusters = linkage(X, method='complete', metric='euclidean')
    plt.figure(figsize=(8, 8))
    row_dendr = dendrogram(row_clusters, orientation='left')
    plt.title(f'Dendrogram ({title})')
    plt.xlabel('Distance')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()

def silhouette_plot(X, y, n_clusters):
    cluster_labels = np.unique(y)
    silhouette_vals = silhouette_samples(X, y, metric='euclidean')
    silhouette_avg = np.mean(silhouette_vals)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

plot_clusters(X_pca, y_km, 'K-means Clustering (PCA-reduced)')
plot_dbscan(X_pca, y_db, 'DBSCAN Clustering (PCA-reduced)')
plot_dendrogram(X_pca, 'Agglomerative Clustering (PCA-reduced)')

for n_clusters in [3, 4, 5]:
    km = KMeans(n_clusters=n_clusters, random_state=0)
    y_km = km.fit_predict(X_pca)
    silhouette_plot(X_pca, y_km, n_clusters)
