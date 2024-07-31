import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def merge_data(gene_filepath, survival_filepath):
    # Load gene expression and survival data
    gene_expression_data = pd.read_csv(gene_filepath, index_col=0)  # Hugo_Symbol as index
    survival_data = pd.read_csv(survival_filepath)

    # Handle duplicate columns
    gene_expression_data = handle_duplicate_columns(gene_expression_data)

    # Transpose gene expression data so that samples are rows and genes are columns
    gene_expression_data_transposed = gene_expression_data.transpose()
    gene_expression_data_transposed.index.name = 'Sample ID'
    gene_expression_data_transposed.reset_index(inplace=True)

    # Merge data using Sample ID
    merged_data = pd.merge(gene_expression_data_transposed, survival_data, on='Sample ID', how='inner')

    # Handle missing values
    merged_data = handle_missing_values(merged_data)

    return merged_data

def handle_duplicate_columns(df):
    # Identify and handle duplicate columns
    cols = pd.Series(df.columns)
    duplicate_cols = cols[cols.duplicated()].unique()

    # Combine duplicate columns by averaging their values
    for col in duplicate_cols:
        cols_to_combine = [c for c in df.columns if c == col]
        df[col] = df[cols_to_combine].mean(axis=1)
        df = df.drop(columns=cols_to_combine[1:])  # Drop all but the first column

    return df

def standardize_data(df):
    # Ensure all column names are strings
    df.columns = df.columns.astype(str)
    
    # Identify features (excluding non-gene columns)
    features = [col for col in df.columns if col not in ['Sample ID', 'Overall Survival (Months)', 'Overall Survival Status']]
    
    # Check for empty feature list
    if not features:
        raise ValueError("No valid features found for standardization. Please check column names.")
    
    # Standardize gene expression values
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Reduce dimensionality to handle collinearity
    pca = PCA(n_components=min(100, len(features)))  # Adjust n_components as needed
    df_pca = pd.DataFrame(pca.fit_transform(df[features]), columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    # Replace original features with PCA components
    df = df.drop(columns=features)
    df = pd.concat([df, df_pca], axis=1)
    
    return df

def handle_missing_values(df):
    # Drop rows with missing values
    df = df.dropna()
    return df

# Cox function for each gene
def coxphf_p_value(gene, data):
    cph = CoxPHFitter()

    if gene in data.columns:
        df = data[['Overall Survival (Months)', 'Overall Survival Status', gene]].copy()
        df.columns = ['duration', 'event', 'gene_expression']
        df = handle_missing_values(df)  # Ensure no missing values in the subset
        try:
            cph.fit(df, duration_col='duration', event_col='event')
            return cph.summary.loc['gene_expression', 'p']
        except Exception as e:
            print(f"Error fitting Cox model for gene {gene}: {e}")
            return None
    else:
        print(f"Gene {gene} not found in the data columns.")
        return None

# Conducts Cox feature selection
def feature_selection(gene_filepath, survival_filepath, output_file):
    data = merge_data(gene_filepath, survival_filepath)

    # Standardize and reduce dimensionality of gene expression data
    data = standardize_data(data)

    # Apply Cox model to each gene
    p_values = {}

    for gene in data.columns[1:]:  # Skip the 'Sample ID' column
        if pd.notna(gene) and gene != '':
            p_value = coxphf_p_value(gene, data)
            if p_value is not None:
                p_values[gene] = p_value
        else:
            print(f"Skipping gene: {gene} (Does not exist or is blank)")
    
    # Select top 100 genes with smallest p-values
    
    top_100_cox = sorted(p_values, key=p_values.get)[:101]
    top_100_cox = top_100_cox[1:101]
    pd.DataFrame(top_100_cox, columns=['Gene']).to_csv(output_file, index=False)
    print(f"Top 100 genes with smallest p-values saved to {output_file}")
