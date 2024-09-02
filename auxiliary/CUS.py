import pandas as pd        
from sklearn.cluster import KMeans  
from collections import Counter   


def CUS(
    c_data, 
    var, 
    target_col, 
    k=10, 
    minority_percentage=0.5, 
    percentage=0.05, 
    minority=False, 
    tw=False, 
    year=2016, 
    random_state=4444
):
    # Make a copy of the data to avoid modifying the original dataset
    data = c_data.copy()
    X = data[var]
    y = data[target_col]
    
    # Identify the majority and minority classes based on value counts
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()
    
    # Separate majority and minority class instances
    majority_df = data[data[target_col] == majority_class]
    minority_df = data[data[target_col] == minority_class]
    
    if not minority:
        # Apply K-means clustering to majority class data
        X_majority = majority_df[var]
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        majority_df['cluster'] = kmeans.fit_predict(X_majority)
        
        # Under-sample majority class data within each cluster
        sampled_majority_df = pd.DataFrame()
        for cluster in range(k):
            cluster_df = majority_df[majority_df['cluster'] == cluster]
            sampled_cluster_df = cluster_df.sample(frac=percentage, random_state=random_state)
            sampled_majority_df = pd.concat([sampled_majority_df, sampled_cluster_df])
        
        # Drop the 'cluster' column after sampling
        sampled_majority_df = sampled_majority_df.drop('cluster', axis=1)
        
        # Combine the sampled majority class with the minority class to form a balanced dataset
        balanced_df = pd.concat([sampled_majority_df, minority_df])
    else:
        # If 'minority' is True, apply K-means clustering to both majority and minority class data
        X_majority = majority_df[var]
        X_minority = minority_df[var]
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        majority_df['cluster'] = kmeans.fit_predict(X_majority)
        minority_df['cluster'] = kmeans.fit_predict(X_minority)
        
        # Under-sample both majority and minority class data within each cluster
        sampled_majority_df = pd.DataFrame()
        sampled_minority_df = pd.DataFrame()
        for cluster in range(k):
            cluster_df_major = majority_df[majority_df['cluster'] == cluster]
            sampled_cluster_df_major = cluster_df_major.sample(frac=percentage, random_state=random_state)
            sampled_majority_df = pd.concat([sampled_majority_df, sampled_cluster_df_major])

            cluster_df_minor = minority_df[minority_df['cluster'] == cluster]
            sampled_cluster_df_minor = cluster_df_minor.sample(frac=minority_percentage, random_state=random_state)
            sampled_minority_df = pd.concat([sampled_minority_df, sampled_cluster_df_minor])
        
        # Drop the 'cluster' column after sampling
        sampled_majority_df = sampled_majority_df.drop('cluster', axis=1)
        sampled_minority_df = sampled_minority_df.drop('cluster', axis=1)
        
        # Combine the sampled majority and minority class data to form a balanced dataset
        balanced_df = pd.concat([sampled_majority_df, sampled_minority_df])
    
    # Shuffle the dataset to ensure random distribution of instances
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Separate features (X_balanced) and target (y_balanced) for the balanced dataset
    X_balanced = balanced_df[var].values
    y_balanced = balanced_df[target_col].values
    
    # If time-weighted (`tw`) is requested, return the additional time variable `t`
    if tw:
        t = (year - balanced_df['fyear']).values
        return X_balanced, y_balanced, t
    else:
        return X_balanced, y_balanced
