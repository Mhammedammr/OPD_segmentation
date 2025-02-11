import streamlit as st
from src.insight.models import Model
import pandas as pd
import numpy as np
from src.insight import utils
import os
import altair as alt
import io
from datetime import datetime

# Streamlit UI
col1, col2, col3 = st.columns([1, 3, 1])  # Center column is wider
with col2:
    st.image("assets/imgs/andalusia_logo.png", width=375)
    
# st.image("assets/imgs/andalusia_logo.png", width=400)
st.title("🔍 AI-Powered Segmentation: Zero-code Clustering & Analysis")
config = {}
data = pd.DataFrame()

# File upload
st.markdown("### Step 1: Upload Training Data")
uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX):", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension.lower() == "csv":
        data = pd.read_csv(uploaded_file)
    elif file_extension.lower() in ["xls", "xlsx"]:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        if len(sheet_names) == 1:
            data = pd.read_excel(uploaded_file)
        else:
            st.write("This Excel file contains multiple sheets. Please select a sheet to load:")
            selected_sheet = st.selectbox("Select a sheet", sheet_names)
            data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    # Data Preview
    st.markdown("### Step 2: Data Overview")
    st.write("Here is a preview of your uploaded data:")
    st.dataframe(data.head())
    
    num_des_analysis, cat_des_analysis, missing_percentage, dups_percentage, u_cols = utils.descriptive_analysis(data)
    with st.expander("Overview", expanded=True):
        st.write(
            """
        Explore your dataset's key characteristics, including numerical summaries, categorical insights, data types, missing values, and duplicates.
        Use the sections below for detailed breakdowns and visualizations.
        """
        )
        st.write(
            f"**Number of Rows:** {data.shape[0]} | **Number of Columns:** {data.shape[1]}"
        )

    # Numerical Description
    with st.expander("Numerical Description"):
        st.write(
            """
        A statistical summary of numeric columns, including metrics like mean, standard deviation, minimum, and maximum values.
        These metrics help identify trends, anomalies, and overall data distribution.
        """
        )
        if num_des_analysis is not None:
            st.dataframe(
                num_des_analysis.style.format(precision=2).background_gradient(cmap="coolwarm"),
                use_container_width=True,
            )
        else:
            st.write("The uploaded file doesn't conatin numerical data")
    
    # Categorical Description
    with st.expander("Categorical Description"):
        st.write(
            """
        A summary of categorical columns showing the count and unique values for each category.
        This helps you understand the variety and frequency of categories in your dataset.
        """
        )
        if cat_des_analysis is not None:
            st.dataframe(
                cat_des_analysis.style.format(precision=2).background_gradient(cmap="coolwarm"),
                use_container_width=True,
            )
        else:
            st.write("The uploaded file doesn't conatin categorical data")

    # Missing Data
    with st.expander("Missing Values (% per Column)"):
        st.write(
            """
        This chart highlights the percentage of missing values in each column. Missing data should be addressed to maintain analysis and modeling accuracy.
        """
        )

        missing = missing_percentage.reset_index()  # Convert Series to DataFrame
        col_index = missing.columns[0]
        missing.columns = [col_index, "missing %"]

        # Ensure 'Missing_Percentage' is numeric
        missing["missing %"] = pd.to_numeric(missing["missing %"], errors="coerce")

        # Drop rows with invalid numeric data
        missing = missing.dropna(subset=["missing %"])

        # Plot using Altair
        chart = (
            alt.Chart(missing)
            .mark_bar()
            .encode(
                x=alt.X(col_index, sort=None),
                y=alt.Y("missing %", title="Missing Percentage (%)"),
                tooltip=[col_index, "missing %"],
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    
    # Duplicate Rows
    with st.expander("Duplicate Rows"):
        st.write(
            f"""
        **Duplicate Rows:** {round(dups_percentage * 100, 2)}%

        Duplicate rows represent repeated records, which may lead to biased analysis. It's recommended to remove them if unnecessary.
        """
        )
    
    # Unique Values
    with st.expander("Unique Values (% per Column)"):
        if isinstance(u_cols, pd.DataFrame):
            u_cols.columns = ["Column", "Unique_Percentage"]

        # Ensure 'Unique_Percentage' is numeric
        u_cols["Unique_Percentage"] = pd.to_numeric(
            u_cols["Unique_Percentage"], errors="coerce"
        )

        # Drop rows with invalid numeric data
        unique = u_cols.dropna(subset=["Unique_Percentage"])

        chart = (
            alt.Chart(unique)
            .mark_bar()
            .encode(
                x=alt.X("Column", sort=None),
                y=alt.Y("Unique_Percentage", title="Unique_Percentage (%)"),
                tooltip=["Column", "Unique_Percentage"],
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
        
    _df = utils.handle_negative(data)
    back_DF = _df.copy()
    
    # Select columns to include
    st.markdown("### Step 3: Select Fetaures For The Clustering")
    columns_to_include = st.multiselect("Select columns to include:", data.columns.tolist())
    features = columns_to_include
    df = back_DF[columns_to_include]
    st.write("Dataset after columns inclusion:", df.head())
    
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
    📖 **Tip**: Use this summary to clean, preprocess, and understand your data better before training models.
    If you encounter high missing values or duplicates, consider cleaning the data for optimal results.
    """
    )
    if 'config' not in st.session_state:
        st.session_state['config'] = {}
    
    # Data Preprocessing
    st.markdown("### Step 4: Data Preprocessing")
    with st.expander(" :broom: Handle Missing Data"):
        clean = st.selectbox(
            "Choose how to handle missing data:",
            ["Remove Missing Data", "Impute with Mean", "Impute with Median", "Impute with Mode"],
            help="Select whether to remove or impute missing values.",
        )
    config["clean"] = clean
    st.session_state['config'] = config
    
    
    with st.expander("📉 Handle Outliers"):
        # Create two columns for horizontal alignment
        col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
        with col1:
            st.write(
                """
            Removes or adjusts extreme data points that can skew results. Choose from:
            - **Don't Remove**: Keep all data points, including outliers.
            - **Use IQR**: Remove data points that are too far from the usual range of most data. For example, it’s like
              ignoring the very low and very high prices in a list of product sales to focus on the middle range.
            - **Isolation Forest**: Use machine learning algorithms to handle outliers.
            """
            )

        with col2:
            # st.image("assets/imgs/outliers.png", use_column_width=True)
            st.image("assets/imgs/outliers.png", use_container_width=True)
        config["outlier"] = st.selectbox(
            "How would you like to handle outliers?",
            ["Don't Remove", "Use IQR", "Use Isolation Forest"],
            help="Select an outlier handling technique.",
        )

        iqr_multiplier, contamination = None, None
        if config["outlier"] == "Use IQR":
            config["multiplier"] = st.slider("Select IQR Multiplier:", 1.0, 3.0, 1.5)
            config["contamination"] = None
            
        elif config["outlier"] == "Use Isolation Forest":
            config["contamination"] = st.slider("Set Contamination Level for Isolation Forest:", 0.01, 0.5, 0.1)
            config["multiplier"] = None
            
        else:
            config["contamination"] = None
            config["multiplier"] = None
            
    
    st.session_state['config'] = config
    
    # Initialize session state for cleaned_df, outlier_df, and re_added_rows
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = pd.DataFrame()
    if 'outlier_df' not in st.session_state:
        st.session_state['outlier_df'] = pd.DataFrame()
    if 're_added_rows' not in st.session_state:
        st.session_state['re_added_rows'] = []



    # Preprocess data
    if st.button("Preprocess Data"):
        df = utils.missing_adv(df, st.session_state['config'])
        
        cleaned_df, outlier_df = utils.remove_outliers(df, st.session_state['config'])
                
        st.session_state['cleaned_df'] = cleaned_df
        st.session_state['outlier_df'] = outlier_df
        
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("Cleaned Dataframe")
        st.dataframe(st.session_state['cleaned_df'])
    with col2:
        st.write("Outlier Dataframe")
        st.dataframe(st.session_state['outlier_df'])

    # Re-add outlier rows
    st.markdown("### Step 5: Re-add Outlier Rows")

    # Get the current outlier_df
    outlier_df = st.session_state['outlier_df']

    # Ensure selected rows exist in the current outlier_df
    valid_rows = [row for row in st.session_state['re_added_rows'] if row in outlier_df.index]
    st.session_state['re_added_rows'] = valid_rows

    # Multiselect for re-adding rows
    selected_rows = st.multiselect(
        "Select rows from the outlier dataset to re-add to the cleaned DataFrame:",
        options=outlier_df.index.tolist(),
        default=st.session_state['re_added_rows'],  # Default to previously selected rows
        format_func=lambda x: f"Row {x}"
    )

    previously_added = set(st.session_state['re_added_rows'])
    currently_selected = set(selected_rows)
    
    user_responsibility = st.checkbox(
        "I take responsibility for re-adding these outlier rows to the cleaned dataset."
    )

    if user_responsibility and len(selected_rows) != 0:
        rows_to_add = currently_selected - previously_added
        rows_to_remove = previously_added - currently_selected

        if rows_to_add or rows_to_remove:
            cleaned_df = st.session_state['cleaned_df'].copy()
            if rows_to_add:
                rows_to_append = outlier_df.loc[list(rows_to_add)]
                cleaned_df = pd.concat([cleaned_df, rows_to_append], ignore_index=True)
                st.success(f"Added {len(rows_to_add)} newly selected row(s).")
            if rows_to_remove:
                rows_to_remove_data = outlier_df.loc[list(rows_to_remove)]
                mask = ~cleaned_df.isin(rows_to_remove_data.to_dict('list')).all(axis=1)
                cleaned_df = cleaned_df[mask]
                st.info(f"Removed {len(rows_to_remove)} unselected row(s).")
            
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['re_added_rows'] = list(currently_selected)
            st.write("Updated Cleaned DataFrame:")
            st.write(cleaned_df.tail(5))
        elif not currently_selected:
            st.warning("Please select at least one row to re-add.")
        else:
            st.info("No changes in selection.")
    else:
        if previously_added:
            cleaned_df = st.session_state['cleaned_df'].copy()
            rows_to_remove_data = outlier_df.loc[list(previously_added)]
            mask = ~cleaned_df.isin(rows_to_remove_data.to_dict('list')).all(axis=1)
            cleaned_df = cleaned_df[mask]
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['re_added_rows'] = []
            st.warning("Responsibility checkbox unchecked. All previously re-added rows have been removed.")
            st.write("Updated Cleaned DataFrame:")
            st.write(cleaned_df.tail(5))
        else:
            st.info("No rows have been re-added.")
    
    # Plot graphs
    if st.button("Plot Graphs"):
        dist_plots = utils.plot_numeric_features(st.session_state['cleaned_df'])
        for dist_plot in dist_plots:
            st.plotly_chart(dist_plot)
    
    
    st.markdown("### Step 6: Select Clustering Technique That Best Fit the Data")
    # Choose clustering technique
    with st.expander("📖 Help Choose The Best Clustering Technique"):
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### K-Means Clustering")
            st.markdown("""
            **Advantages**:
            - Simple and fast for large datasets.
            - Works well for spherical clusters.
            - Easy to interpret and implement.

            **Disadvantages**:
            - Requires the number of clusters (`k`) to be specified.
            - Sensitive to initial centroid placement.
            - Struggles with non-spherical or overlapping clusters.
            """)

        with col2:
            st.markdown("### Gaussian Mixture Models (GMM)")
            st.markdown("""
            **Advantages**:
            - Handles overlapping clusters.
            - Works well for elliptical or Gaussian-shaped clusters.
            - Provides probabilistic cluster assignments.

            **Disadvantages**:
            - Computationally more expensive than K-Means.
            - Requires the number of components (`n_components`) to be specified.
            """)

        # Key Differences Table
        st.markdown("### Key Differences")
        st.markdown("""
        | **Aspect**              | **K-Means**                          | **GMM**                              |
        |--------------------------|--------------------------------------|--------------------------------------|
        | **Cluster Shape**        | Spherical                           | Elliptical or Gaussian               |
        | **Cluster Overlap**      | No overlap (hard clustering)        | Allows overlap (soft clustering)     |
        | **Data Distribution**    | No specific distribution assumed    | Assumes Gaussian distribution        |
        | **Speed**                | Faster                              | Slower                               |
        | **Use Case**             | Simple, well-separated clusters     | Complex, overlapping clusters        |
        """)


    if 'data_with_clusters' not in st.session_state:
        st.session_state['data_with_clusters'] = None
    
    # Clustering Technique Selection
    tech = st.selectbox(
        "Select Clustering Technique:",
        ["KMeans", "GMM"],
        help="Select a Clustering Technique."
    )
    st.session_state['config']["alg"] = tech
    
    st.session_state['config']["model_kw"] = {}
    if st.session_state['config']["alg"] == "KMeans":
        k_options = list(range(2, 10)) + [(-1, "Auto Select Best K")]
        k_selection = st.selectbox("Number of Clusters (k):", k_options, 1)
        st.session_state['config']['model_kw']['n_clusters'] = k_selection if not isinstance(k_selection, tuple) else -1
        if isinstance(k_selection, tuple):
            st.info("Silhouette Analysis will automatically select the best number of clusters.")
    
    elif st.session_state['config']["alg"] == "GMM":
        st.session_state['config']['model_kw']['n_components'] = st.slider("Number of Clusters:", 2, 10, 3)

    if 'data_with_clusters' not in st.session_state:
        st.session_state['data_with_clusters'] = None

    # Perform clustering
    if st.session_state['cleaned_df'].shape[1] > 0:
        model = Model(st.session_state['config'])
        model.train(st.session_state['cleaned_df'])
        data_with_clusters, additional_info = model.process_cluster(st.session_state['cleaned_df'])
        
        if st.session_state['config']["alg"] in ["KMeans", "GMM"] and st.session_state['config']['model_kw'].get('n_clusters', st.session_state['config']['model_kw'].get('n_components')) == 3:
            data_with_clusters = utils.rename_clusters(data_with_clusters)
        
        if len(df.columns) == 2:
            st.write("##### Clustering Results:")
            scatter_plot = utils.scatter_plots(data_with_clusters)
            st.plotly_chart(scatter_plot)
        else:
            columns_to_include = st.multiselect("Select columns to perform 2D scatter plot:", df.columns.tolist())
            columns_to_include.append("cluster")
            _df = data_with_clusters[columns_to_include]
            st.write("Dataset after column exclusion:", _df.head(2))
            if st.button("Plot") and (_df.shape[1] == 2 or _df.shape[1] == 3):
                st.write("### Clustering Results:")
                scatter_plot = utils.scatter_plots(_df)
                st.plotly_chart(scatter_plot)
            else:
                st.info("Please Select columns")

        if additional_info and additional_info.get('best_k'):
            st.success(f"Best number of clusters (k) identified: {additional_info['best_k']}")
        
        fig = utils.cluster_dist(data_with_clusters)
        st.plotly_chart(fig)
        
        st.markdown("### Step 7: Description of Features across Clusters")
        descs, name_list = utils.clusters_analysis(data_with_clusters)
        for i in range(0, data_with_clusters.shape[1] - 1):
            st.text(f"Description of {name_list[i]} across Clusters")
            descs[i]["count"] = pd.to_numeric(descs[i]["count"], errors="coerce")
            st.table(descs[i].sort_values(by="count", ascending=True).applymap(lambda x: f"{x:.2f}"))
    
        st.session_state['data_with_clusters'] = data_with_clusters
        
        
    if st.session_state['data_with_clusters'] is not None:
        # Convert DataFrame to Excel
        excel_buffer = io.BytesIO()
        st.session_state['data_with_clusters'].to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        # Create download button
        st.download_button(
            label="Download Excel File",
            data=excel_buffer,
            file_name="Data_with_Cluster.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        tech = st.session_state['clustering_technique']
        utils.tracking(st.session_state['data_with_clusters'], features, st.session_state['config'])
    else:
        st.warning("No clustered data available to save. Please perform clustering first.")
        
