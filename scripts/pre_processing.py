import pandas as pd
import numpy as np

# Function to load gene expression data
def load_gene_expression_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data

# Function to filter out low-expression genes
def filter_low_expression_genes(data, threshold=1.0):
    # Filter out genes with mean expression below the threshold
    data_filtered = data.loc[:, (data.mean(axis=0) >= threshold).values]
    return data_filtered

# Function to preprocess gene expression data
def preprocess_gene_expression_data(file_path, output_gene_file, threshold):
    # Load the data
    data = load_gene_expression_data(file_path)
    
    # Set the gene symbols as the index
    data.set_index('Hugo_Symbol', inplace=True)
    
    # Filter out low-expression genes
    data_filtered = filter_low_expression_genes(data, threshold)
    
    # Save the preprocessed data
    data_filtered.to_csv(output_gene_file)
    print(f"Processed gene expression data saved to {output_gene_file}")


# Function to extract the survival data (time to event) and event status from the clinical file.
def preprocess_clinical_data(file_path, output_clinical_file):
    # Load clinical data
    data = pd.read_csv(file_path, sep='\t')

    # Extract relevant columns
    survival_data = data[['Sample ID', 'Overall Survival (Months)', 'Overall Survival Status']]

    # Handle missing values
    survival_data = survival_data.dropna()

    # Process event status
    survival_data['Overall Survival Status'] = survival_data['Overall Survival Status'].apply(
        lambda x: 1 if 'DECEASED' in x.upper() else 0
    )

    # Save the processed data
    survival_data.to_csv(output_clinical_file, index=False)
    print(f"Processed survival data saved to {output_clinical_file}")

