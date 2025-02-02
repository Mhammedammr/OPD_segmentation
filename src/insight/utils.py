import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def descriptive_analysis(df):
    """
    Visualize descriptive statistical analysis, data types and missing, duplicates and unique values percentage for each features.
    
    Returns:
        num_des_analysis: descriptive stat analysis on numerical feature.
        cat_des_analysis: descriptive stat analysis on categorcal feature.
        d_types: data types of each feature.
        missing_percentage: missing values perc for each feature.
        dups_percentage: duplicate records perc.
        unq_percentage: unique values perc for each feature.
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    numerical_columns = df.select_dtypes(include=["number"]).columns
    
    if len(numerical_columns) > 0:
        num_des_analysis = df.describe().T
    else:
        num_des_analysis = "No numerical columns in the DataFrame."
    if len(categorical_columns) > 0:
        cat_des_analysis = df.describe(include=["object", "category"]).T
    else:
        cat_des_analysis = "No categorical columns in the DataFrame."
    d_types = pd.DataFrame(df.dtypes, columns=["type"])
    missing_percentage = pd.DataFrame(
        (df.isna().sum() / len(df)) * 100, columns=["missing %"]
    ).round(2)
    dups_percentage = (len(df[df.duplicated()]) / len(df)) * 100
    unq_percentage = unique_percentage(df)

    return (
        num_des_analysis,
        cat_des_analysis,
        d_types,
        missing_percentage,
        dups_percentage,
        unq_percentage,
    )


def unique_percentage(df):
    """
    Gets unique record percentage in a dataframe's features.
    
    Returns:
            Dataframe where each column name maps for unique values percentages
    """
    return (
        pd.DataFrame(
            {
                "Column": df.columns,
                "Unique_Percentage": [
                    (df[col].nunique() / len(df[col])) * 100 for col in df.columns
                ],
            }
        )
        .sort_values(by="Unique_Percentage", ascending=False)
        .reset_index(drop=True)
    )


def missing_adv(_df, cfg):
    """
    Handling missing values with the tech that user selects.
    
    Returns:
        df: DataFrame without null values.
    """
    df = _df.copy()

    if cfg["clean"] == "Remove Missing Data":
        df.dropna(inplace=True)
        return df

    elif cfg["clean"] == "Impute with Mean":
        df = imputer(df, "mean")

    elif cfg["clean"] == "Impute with Median":
        df = imputer(df, "median")
        
    elif cfg["clean"] == "Impute with Mode":
        df = imputer(df, "most_frequent")

    return df


def imputer(_df, strategy="mean"):
    """
    Impute missing values using the mean, median and mode tech with numeric features and using mode tech with categorical features.
    
    Returns:
        df: DataFrame without null values.
    """
    df = _df.copy()  # Avoid modifying the original dataframe

    numeric_features = df.select_dtypes(include=np.number).columns
    categorical_features = df.select_dtypes(include=object).columns

    # Handle empty categorical columns
    if len(categorical_features) > 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    # Handle empty numeric columns
    if len(numeric_features) > 0:
        numeric_imputer = SimpleImputer(strategy=strategy)
        df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    return df


def IQR(_df, lower_bound=0.25, upper_bound=0.75, multiplier=None):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Returns:
        clean_df: DataFrame without outliers.
        outlier_df: DataFrame containing only the outliers.
    """
    df = _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    sub_df = df[numeric_cols]

    # Calculate IQR
    q1 = sub_df[numeric_cols].quantile(lower_bound)
    q3 = sub_df[numeric_cols].quantile(upper_bound)
    iqr = q3 - q1

    # Identify outliers
    outlier_mask = ((sub_df < (q1 - multiplier * iqr)) | (sub_df > (q3 + multiplier * iqr))).any(axis=1)
    
    # Split into clean and outlier DataFrames
    clean_df = df[~outlier_mask]  # Rows without outliers
    outlier_df = df[outlier_mask]  # Rows with outliers

    return clean_df, outlier_df


def IF(_df, contamination=None):
    """
    Remove outliers using the Isolation Forest method.
    
    Returns:
        clean_df: DataFrame without outliers.
        outlier_df: DataFrame containing only the outliers.
    """
    isolation_forest = IsolationForest(contamination=contamination)
    df = _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]

    num_df = df[numeric_cols]
    outlier_pred = isolation_forest.fit_predict(num_df)

    # Split into clean and outlier DataFrames
    clean_df = df[outlier_pred == 1]  # Rows without outliers
    outlier_df = df[outlier_pred == -1]  # Rows with outliers

    return clean_df, outlier_df


def remove_outliers(_df, method=None, iqr_multiplier=None, contamination=None):
    """
    Remove outliers using the selected tech by the user.
    
    Returns:
        clean_df: DataFrame without outliers.
        outlier_df: DataFrame containing only the outliers.
    """
    if method == "Use IQR":
        cleaned_df, outlier_df = IQR(_df, multiplier=iqr_multiplier)
    elif method == "Use Isolation Forest":
        cleaned_df, outlier_df = IF(_df, contamination=contamination)
    else:
        cleaned_df, outlier_df = _df, pd.DataFrame(columns=_df.columns)
        
    return cleaned_df, outlier_df


def plot_numeric_features(df):
    """
    Plots the distribution of each numeric feature in a dataframe using interactive Plotly histograms.
    Helps users choose appropriate techniques based on data distribution.
    
    Returns:
        plots: list of Plotly figures that can be easily visualized in Streamlit.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [col for col in numeric_cols if df[col].nunique() >= 10]

    plots = []
    for col in valid_cols:
        # Compute statistics
        skewness = stats.skew(df[col].dropna())
        _, p_value = stats.normaltest(df[col].dropna())

        # Create interactive histogram
        fig = px.histogram(
            df, x=col, nbins=30,
            title=f"{col} Distribution<br>Skewness: {skewness:.2f}, Normality p-value: {p_value:.4f}",
            color_discrete_sequence=["steelblue"]
        )

        # Add KDE (density) line
        fig.add_trace(
            go.Scatter(
                x=np.sort(df[col].dropna()),
                y=stats.gaussian_kde(df[col].dropna())(np.sort(df[col].dropna())) * len(df[col].dropna()) * (df[col].max() - df[col].min()) / 30,
                mode="lines",
                line=dict(color="red"),
                name="Density"
            )
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frequency",
            template="plotly_white"
        )

        plots.append(fig)

    excluded_cols = set(numeric_cols) - set(valid_cols)
    if excluded_cols:
        print(f"Excluded columns (less than 10 unique values): {', '.join(excluded_cols)}")

    return plots


def cluster_dist(pca_data: pd.DataFrame):
    """
    Distrbution of samples across cluster with percentage.
    
    Returns:
        fig: histogram of the distrbutions.
    """
    cluster_percentage = (
        pca_data["cluster"].value_counts(normalize=True) * 100
    ).reset_index()
    cluster_percentage.columns = ["Cluster", "Percentage"]
    cluster_percentage.sort_values(by="Cluster", inplace=True)

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Percentage", y="Cluster", data=cluster_percentage, orient="h")

    # Adding percentages on the bars
    for index, value in enumerate(cluster_percentage["Percentage"]):
        ax.text(value + 0.5, index, f"{value:.2f}%")

    plt.title("Distribution Across Clusters", fontsize=14)
    plt.xticks(ticks=np.arange(0, 50, 5))
    plt.xlabel("Percentage (%)")

    # Show the plot
    return fig


def clusters_analysis(df_with_cluster):
    """
    Describe the data across features and clusters to gain more insights.
    
    Args:
        df_with_cluster: DataFrame containing features and a 'cluster' column
    
    Returns:
        description: table with statistical values, features in first row
    """
    df_without_cluster = df_with_cluster.drop(columns=["cluster"])
    
    numeric_df = df_without_cluster.select_dtypes(include=[np.number])
    features_names = numeric_df.columns
    
    desc = []
    name_list = []
    # Get the description
    description = df_with_cluster.groupby("cluster").describe()
    for feature in features_names:
        name_list.append(feature)
        desc.append(description[feature])
    return desc, name_list


def scatter_plots(df):
    """
    Create scatterplot after clustering to visualize the clusters more to the user to take decisions.
    
    Returns:
        fig: figure to make it easy for streamlit to visualize.
    """
    coln_1 = df.columns[0]
    coln_2 = df.columns[1]
    
    # Create a scatter plot using Plotly Express
    fig = px.scatter(df, x=coln_1, y=coln_2, color="cluster",
                     labels={coln_1: coln_1, coln_2: coln_2},
                     color_continuous_scale=px.colors.sequential.Plasma)
    
    # Update marker size and edge color
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    
    # Return the figure to be shown in Streamlit
    return fig


def rename_clusters(df):
    """
    Map cluster names to logical names in case number of cluster is 3 based on the count of samples in the clusters.
    
    Returns:
            df: dataframe with cluster column with logical names as values
    """
    # Count the number of instances in each cluster
    cluster_counts = df["cluster"].value_counts()

    # Map clusters to custom labels based on instance counts
    sorted_clusters = cluster_counts.sort_values().index  # Sort clusters by size
    cluster_mapping = {sorted_clusters[0]: "High", sorted_clusters[1]: "Medium", sorted_clusters[2]: "Low"}
    
    # Apply the mapping to rename clusters
    df["cluster"] = df["cluster"].map(cluster_mapping)
    
    return df