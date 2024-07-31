import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_data(gene_filepath, top_genes_filepath):
    gene_expression_data = pd.read_csv(gene_filepath, index_col=0)
    top_genes = pd.read_csv(top_genes_filepath)
    top_genes_list = top_genes['Gene'].tolist()
    return gene_expression_data[top_genes_list]

def determine_optimal_clusters(data):
    inertia = []
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')

    plt.tight_layout()
    plt.show()

    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    return cluster_labels

def save_clustering_results(data, cluster_labels, output_filepath):
    data['Cluster'] = cluster_labels
    data.to_csv(output_filepath, index=True)
    print(f"Clustering results saved to {output_filepath}")
