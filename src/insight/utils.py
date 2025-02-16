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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import plotly.express as px
import plotly.graph_objects as go

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


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
    if len(df) == 0:
        raise ValueError("The DataFrame is completly empty.")

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
    missing_percentage = pd.DataFrame(
        (df.isna().sum() / len(df)) * 100, columns=["missing %"]
    ).round(2)
    dups_percentage = (len(df[df.duplicated()]) / len(df)) * 100
    unq_percentage = unique_percentage(df)

    return (
        num_des_analysis,
        cat_des_analysis,
        missing_percentage,
        dups_percentage,
        unq_percentage,
    )


def handle_negative(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) == 0:
        # Check for rows where any of the selected columns have negative values
        rows_with_negatives = (df < 0).any(axis=1)
        
        # Remove rows with negative values in the selected columns
        df_cleaned = df[~rows_with_negatives]
        return df_cleaned

    else:
        raise ValueError("can not remove negative values from categorical columns, please select numeric columns")

                    
                    
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
        df.replace({"None": np.nan}, inplace=True)
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

    df.replace({"None": np.nan}, inplace=True)
    
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
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
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


def remove_outliers(_df, config):
    """
    Remove outliers using the selected tech by the user.
    
    Returns:
        clean_df: DataFrame without outliers.
        outlier_df: DataFrame containing only the outliers.
    """
    if config["outlier"] == "Use IQR":
        cleaned_df, outlier_df = IQR(_df, multiplier=config["multiplier"])
    elif config["outlier"] == "Use Isolation Forest":
        cleaned_df, outlier_df = IF(_df, contamination=config["contamination"])
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
        # Create interactive histogram
        fig = px.histogram(
            df, x=col, nbins=30,
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


def cluster_dist(df: pd.DataFrame):
    """
    Distribution of samples across clusters with percentage.
    
    Returns:
        fig: Interactive histogram of the distributions using Plotly.
    """
    # Calculate cluster percentages
    cluster_percentage = (
        df["cluster"].value_counts(normalize=True) * 100
    ).reset_index()
    cluster_percentage.columns = ["Cluster", "Percentage"]
    cluster_percentage.sort_values(by="Percentage", inplace=True, ascending=False)

    # Create an interactive bar plot using Plotly
    fig = px.bar(
        cluster_percentage,
        x="Percentage",
        y="Cluster",
        orientation="h",
        text="Percentage",
        labels={"Percentage": "Percentage (%)", "Cluster": "Cluster"},
        title="Distribution Across Clusters",
    )

    # Customize the layout
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")  # Add percentage labels
    fig.update_layout(
        xaxis_title="Percentage (%)",
        yaxis_title="Cluster",
        xaxis=dict(tickvals=list(range(0, 55, 5))),  # Set x-axis ticks
        showlegend=False,
        template="plotly_white",  # Use a clean template
    )

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
    
    cluster_order = df["cluster"].value_counts(normalize=True).sort_values(ascending=True)
    # Create a scatter plot using Plotly Express
    fig = px.scatter(df, x=coln_1, y=coln_2, color="cluster",
                     labels={coln_1: coln_1, coln_2: coln_2},
                     color_continuous_scale=px.colors.sequential.Plasma,
                     category_orders={"cluster": cluster_order.to_dict()})
    
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


def tracking(features, config, metrics):
    # Compute evaluation metrics
    silhouette = metrics["silhouette_score"]
    davies_bouldin = metrics["davies_bouldin_index"]
    calinski_harabasz = metrics["calinski_harabasz_index"]

    # Log parameters
    mlflow.log_param("technique", config["alg"])
    
    if config["alg"] == "KMeans":
        mlflow.log_param("n_clusters", config["model_kw"]["n_clusters"])
    elif config["alg"] == "GMM":
        mlflow.log_param("n_clusters", config["model_kw"]["n_components"])
    
    mlflow.log_param("features", features)
    mlflow.log_param("missing_technique", config["clean"])
    mlflow.log_param("outlier_tech", config["outlier"])

    if config["contamination"] is not None:
        mlflow.log_param("contamination", config["contamination"])
    elif config["multiplier"] is not None:
        mlflow.log_param("iqr_multiplier", config["multiplier"])

    # Log metrics
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.log_metric("davies_bouldin_index", davies_bouldin)
    mlflow.log_metric("calinski_harabasz_index", calinski_harabasz)