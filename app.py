import streamlit as st
from src.insight.models import Model
import pandas as pd
import numpy as np
from src.insight import utils


# Streamlit UI
st.title("OPD Segmentation App")
config = {}
data = pd.DataFrame()

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX):", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension.lower() == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension.lower() in ["xls", "xlsx"]:
        # Read the Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        
        # Get the list of sheet names
        sheet_names = excel_file.sheet_names
        
        # If there is only one sheet, read it directly
        if len(sheet_names) == 1:
            data = pd.read_excel(uploaded_file)
        else:
            # If there are multiple sheets, prompt the user to select one
            st.write("This Excel file contains multiple sheets. Please select a sheet to load:")
            selected_sheet = st.selectbox("Select a sheet", sheet_names)
            data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    st.header("Dataset Preview")
    st.write(data.head())
    
    (
        num_des_analysis,
        cat_des_analysis,
        d_types,
        missing_percentage,
        dups_percentage,
        u_cols,
    ) = utils.descriptive_analysis(data)
    
    st.subheader("Numerical Description")
    st.write(num_des_analysis)
    
    st.subheader("Categorical Description")
    st.write(cat_des_analysis)
    
    st.subheader("DataFrame Types")
    st.write(d_types)
    
    st.subheader("Missing per Column %")
    st.write(missing_percentage)
    
    st.subheader(f"Duplicates {round(dups_percentage * 100, 2)} %")
    # st.write(dups_percentage)
    
    st.subheader("Unique %")
    st.write(u_cols)
    
    
    if "data" in locals():
        
        back_DF = data.copy()
        
        # Select columns to include
        st.write("### Select columns to continue")
        columns_to_include = st.multiselect("Select columns to include:", data.columns.tolist())
        df = back_DF[columns_to_include]
        st.write("Dataset after columns inclusion:", df.head())
        
        
        # Handle missing values
        missing_handling = st.selectbox("Select Missing Values Handling Method:", ["Remove Missing Data", "Impute with Mean", "Impute with Median",
                                                                                   "Impute with Mode"])
        config["clean"] = missing_handling
        
        # Handle outliers
        cleaned_df = pd.DataFrame()
        outlier_df = pd.DataFrame()
        
        outlier_handling = st.selectbox("Select Outlier Handling Method:", ["None", "Use IQR", "Use Isolation Forest"])
        config["outlier"] = {"method": outlier_handling}
        iqr_multiplier, contamination = None, None
        if outlier_handling == "Use IQR":
            iqr_multiplier = st.slider("Select IQR Multiplier:", 1.0, 3.0, 1.5)
        if outlier_handling == "Use Isolation Forest":
            contamination = st.slider("Set Contamination Level for Isolation Forest:", 0.01, 0.5, 0.1)

        if 'cleaned_df' not in st.session_state:
            st.session_state['cleaned_df'] = None
            
        if 'outlier_df' not in st.session_state:
            st.session_state['outlier_df'] = pd.DataFrame()

        # Preprocess data
        if st.button("Preprocess Data"):
            df = utils.missing_adv(df, config)
            cleaned_df, outlier_df = utils.remove_outliers(df, method=outlier_handling, iqr_multiplier=iqr_multiplier, contamination=contamination)
            
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['outlier_df'] = outlier_df
            
        st.write("Preprocessed Dataset:", st.session_state['cleaned_df'])
        st.write("Outlier Dataset:", st.session_state['outlier_df'])
        
        
        if 're_added_rows' not in st.session_state:
            st.session_state['re_added_rows'] = []  # Track indices of re-added rows
        
        st.write("### Re-add Outlier Rows")
        selected_rows = st.multiselect(
            "Select rows from the outlier dataset to re-add to the cleaned DataFrame:",
            options=st.session_state['outlier_df'].index.tolist(),  # Allow selection of row indices
            format_func=lambda x: f"Row {x}"  # Display row indices in a user-friendly way
        )
        
        # Get the previously re-added rows
        previously_added = set(st.session_state['re_added_rows'])
        currently_selected = set(selected_rows)
        
        # Add a checkbox for user responsibility
        user_responsibility = st.checkbox(
            "I take responsibility for re-adding these outlier rows to the cleaned dataset."
        )

        if user_responsibility:
            # Find rows to add (newly selected) and rows to remove (unselected)
            rows_to_add = currently_selected - previously_added
            rows_to_remove = previously_added - currently_selected

            if rows_to_add or rows_to_remove:
                # Start with the current cleaned DataFrame
                cleaned_df = st.session_state['cleaned_df'].copy()
                
                # Add newly selected rows
                if rows_to_add:
                    rows_to_append = st.session_state['outlier_df'].loc[list(rows_to_add)]
                    cleaned_df = pd.concat([cleaned_df, rows_to_append], ignore_index=True)
                    st.success(f"Added {len(rows_to_add)} newly selected row(s).")

                    st.session_state['cleaned_df'] = cleaned_df
                    
                # Remove unselected rows
                if rows_to_remove:
                    # Get the index values from outlier_df for rows to remove
                    rows_to_remove_data = st.session_state['outlier_df'].loc[list(rows_to_remove)]
                    
                    # Create a mask to identify these rows in cleaned_df
                    mask = ~cleaned_df.isin(rows_to_remove_data.to_dict('list')).all(axis=1)
                    cleaned_df = cleaned_df[mask]
                    st.info(f"Removed {len(rows_to_remove)} unselected row(s).")
                    st.session_state['cleaned_df'] = cleaned_df
                    
                # Update the session state
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['re_added_rows'] = list(currently_selected)
                
                st.write("Updated Cleaned DataFrame:")
                st.write(cleaned_df.tail(5))
            
            elif not currently_selected:
                st.warning("Please select at least one row to re-add.")
            else:
                st.info("No changes in selection.")

        else:
            # If responsibility checkbox is unchecked, remove all previously re-added rows
            if previously_added:
                cleaned_df = st.session_state['cleaned_df'].copy()
                rows_to_remove_data = st.session_state['outlier_df'].loc[list(previously_added)]
                mask = ~cleaned_df.isin(rows_to_remove_data.to_dict('list')).all(axis=1)
                cleaned_df = cleaned_df[mask]
                
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['re_added_rows'] = []
                
                st.warning("Responsibility checkbox unchecked. All previously re-added rows have been removed.")
                st.write("Updated Cleaned DataFrame:")
                st.write(cleaned_df.tail(5))
            else:
                st.info("No rows have been re-added.")
            
            
        if st.button("Plot Graphs"):
            skew_plots = utils.plot_numeric_features(st.session_state['cleaned_df'])
            for skew_plot in skew_plots:
                st.plotly_chart(skew_plot)
                  
        # Choose clustering technique
        if 'clustering_technique' not in st.session_state:
            st.session_state['clustering_technique'] = None
        clustering_technique = st.selectbox("Select Clustering Technique:", ["KMeans", "DBSCAN", "GMM"])
        st.session_state['clustering_technique'] = clustering_technique
        # Parameters configuration
        config = {"alg": clustering_technique, "model_kw": {}}
        if clustering_technique == "KMeans":
            k_options = list(range(2, 16)) + [(-1, "Auto Select Best K")]
            k_selection = st.selectbox("Number of Clusters (k):", k_options)
            
            # Handle auto selection or specific k
            if isinstance(k_selection, tuple):  # Auto select
                config['model_kw']['n_clusters'] = -1
                st.info("Silhouette Analysis will automatically select the best number of clusters.")
            else:
                config['model_kw']['n_clusters'] = k_selection
        
        elif clustering_technique == "DBSCAN":
            config['model_kw']['eps'] = st.slider("Epsilon:", 0.1, 3.0, 0.5)
            config['model_kw']['min_samples'] = st.slider("Minimum Samples:", 1, 10, 5)
        
        elif clustering_technique == "GMM":
            config['model_kw']['n_components'] = st.slider("Number of Components:", 1, 10, 3)

        # Initialize session state for data_with_clusters
        if 'data_with_clusters' not in st.session_state:
            st.session_state['data_with_clusters'] = pd.DataFrame()

        # Perform clustering
        if st.session_state['cleaned_df'].shape[1] > 0:
            model = Model(config)
            model.train(st.session_state['cleaned_df'])
            data_with_clusters, additional_info = model.process_cluster(st.session_state['cleaned_df'])
                
            if st.session_state['clustering_technique'] == "KMeans":
                if config['model_kw']['n_clusters'] == 3:
                    data_with_clusters = utils.Rename_Clusters(data_with_clusters)
            
            if st.session_state['clustering_technique'] == "GMM":
                if config['model_kw']['n_components'] == 3:
                    data_with_clusters = utils.Rename_Clusters(data_with_clusters)
            
            if len(df.columns) == 2:
                st.write("##### Clustering Results:")
                scatter_plot = utils.Scatter_Plots(data_with_clusters)
                # Display the plot in the Streamlit app
                st.plotly_chart(scatter_plot)
            else:
                columns_to_include = st.multiselect("Select columns to preform 2D scatter plot:", df.columns.tolist())
                columns_to_include.append("cluster")
                _df = data_with_clusters[columns_to_include]
                st.write("Dataset after column exclusion:", _df.head())
                # Generate the plot
                if st.button("Plot") and _df.shape[1] == 2:
                    st.write("### Clustering Results:")
                    scatter_plot = utils.Scatter_Plots(_df)
                    # Display the plot in the Streamlit app
                    st.plotly_chart(scatter_plot)
                else:
                    st.info("Please Select columns")

            # Display best k if silhouette analysis was performed
            if additional_info and additional_info.get('best_k'):
                st.success(f"Best number of clusters (k) identified: {additional_info['best_k']}")
                
            fig = utils.cluster_dist(data_with_clusters)
            st.pyplot(fig)
            
            st.write("#### Description of Features across Clusters")
            numeric_df = data_with_clusters.select_dtypes(include=[np.number])
            descs, name_list = utils.clusters_analysis(data_with_clusters)
            features_names = data_with_clusters.columns
            for i in range(0, numeric_df.shape[1] - 1):
                st.text(f"Description of {name_list[i]} across Clusters")
                st.table(descs[i])
        
        # Save Clustered Data in excel sheet
        if st.button("Save Clustered Data"):
            st.session_state['data_with_clusters'].to_excel('/home/ai/Workspace/AmrJr/OPD_segmentation/data/Data_with_Cluster.xlsx', index=False)