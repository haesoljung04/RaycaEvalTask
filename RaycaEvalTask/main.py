# 1 & 2. Data download and Data preparation

from scripts.pre_processing import preprocess_gene_expression_data
from scripts.pre_processing import preprocess_clinical_data

# adjust to your own file path names if needed
input_gene_file_path = 'data/data_mrna_seq_v2_rsem.txt'   
input_clinical_file_path = 'data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv'
output_gene_file='data/processed_gene_expression.csv'
output_clinical_file='data/processed_clinical.csv'
threshold = 1.0

# can also adjust output file name and threshold here
preprocess_gene_expression_data(input_gene_file_path, output_gene_file, threshold)
preprocess_clinical_data(input_clinical_file_path, output_clinical_file)


# 3. Feature Selection
from scripts.feature_selection import feature_selection

# adjust to your own file path names if needed
input_processed_gene = 'data/processed_gene_expression.csv'   
input_processed_clinical = 'data/processed_clinical.csv'
output_top_genes = 'data/top_100_genes.csv'

feature_selection(input_processed_gene, input_processed_clinical, output_top_genes)


#4. Clustering using k-means algorithm
from scripts.clustering import load_data
from scripts.clustering import determine_optimal_clusters
from scripts.clustering import perform_clustering
from scripts.clustering import save_clustering_results

output_cluster = 'cluster_results.csv'      

data = load_data(input_processed_gene, output_top_genes)
optimal_k = determine_optimal_clusters(data)
print(f"Optimal number of clusters: {optimal_k}")
cluster_labels = perform_clustering(data, optimal_k)
save_clustering_results(data, cluster_labels, output_cluster)
