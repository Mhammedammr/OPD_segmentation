# OPD Segmentation App - User Manual

## Introduction
The **OPD Segmentation App** is a user-friendly, web-based application designed to assist healthcare professionals in analyzing and segmenting patient data from Outpatient Departments (**OPD**). The application leverages advanced data preprocessing, outlier detection, and clustering techniques to help users identify patterns and groupings within their datasets. This manual provides step-by-step instructions on how to use the app effectively.

## Key Features
- **Data Upload**: Supports **CSV** and **Excel** file formats.
- **Descriptive Analysis**: Provides insights into numerical and categorical data, missing values, duplicates, and unique values.
- **Data Preprocessing**: Offers options to handle missing values and outliers.
- **Clustering Techniques**: Includes **KMeans** and **Gaussian Mixture Models (GMM)** for segmentation.
- **Visualization**: Generates scatter plots and cluster distribution charts for better interpretation.
- **Export Results**: Allows users to save clustered data for further analysis.

---

## Getting Started
### 1. Accessing the App
- Open the app in your web browser.
- The app will load with a title: **"OPD Segmentation App"**.

### 2. Uploading Your Dataset
- Click on the **"Upload your dataset (CSV or XLSX)"** button.
- Select a CSV or Excel file from your local machine.
- If the file is an Excel workbook with multiple sheets, you will be prompted to select a specific sheet.
- Once uploaded, the app will display a **preview of the dataset**.

---

## Data Analysis and Preprocessing
### 3. Descriptive Analysis
After uploading the dataset, the app automatically performs a descriptive analysis, including:
- **Numerical Description**: Summary statistics for numerical columns.
- **Categorical Description**: Summary statistics for categorical columns.
- **Data Types**: Displays the data types of each column.
- **Missing Values**: Shows the percentage of missing values per column.
- **Duplicates**: Indicates the percentage of duplicate rows.
- **Unique Values**: Displays the percentage of unique values per column.

### 4. Selecting Columns
- Use the **"Select columns to include"** dropdown to choose the columns you want to analyze.
- This columns that will be used then in clustering.

### 5. Handling Missing Values
- Choose a method to handle missing values:
  - **Remove Missing Data**: Drops rows with missing values.
  - **Impute with Mean/Median/Mode**: Replaces missing values with the mean, median, or mode of the column.
- Click **"Preprocess Data"** to apply the selected method.

### 6. Handling Outliers
- Select an outlier handling method:
  - **None**: No outlier handling.
  - **Use IQR**: Removes outliers based on the Interquartile Range (**IQR**). Adjust the IQR multiplier using the slider.
  - **Use Isolation Forest**: Identifies outliers using the **Isolation Forest algorithm**. Adjust the contamination level using the slider.
- Click **"Preprocess Data"** to apply the selected method.
- The app will display the **cleaned dataset** and the **outlier dataset**.

### 7. Re-adding Outlier Rows
If needed, you can re-add specific outlier rows to the cleaned dataset:
- Select the rows from the outlier dataset using the **multiselect dropdown**.
- Check the **"I take responsibility for re-adding these outlier rows"** checkbox.
- The app will update the cleaned dataset with the selected rows.

---

## Clustering and Segmentation
### 8. Choosing a Clustering Technique
- Select a clustering technique from the dropdown list:
  - **KMeans**: Requires the number of clusters (**k**). You can choose a specific **k** or let the app automatically select the best **k** using **silhouette analysis**.
  - **GMM**: Requires the number of components (**clusters**).
- Adjust the parameters as needed.

### 9. Performing Clustering
- Click **"Plot Graphs"** to visualize the **data distribution and skewness**.
- The app will perform clustering and display the results:
  - **Scatter Plot**: Visualizes the clusters in **2D** (if applicable).
  - **Cluster Distribution**: Shows the distribution of data points across clusters.
  - **Feature Description**: Provides a summary of features across clusters.

---

## Saving Results
### 10. Exporting Clustered Data
- Click **"Save Clustered Data"** to export the clustered dataset to an **Excel file**.
- The file will be saved to the **specified path on the server**.

---

## Best Practices
- **Data Quality**: Ensure that selected columns are representitive and related to your task.
- **Outlier Handling**: Carefully review and re-add outlier rows only if they are clinically relevant.
- **Clustering Parameters**: Experiment with different clustering techniques and parameters to achieve the best results.

---

## Troubleshooting
- **Unsupported File Format**: Ensure your file is in **CSV** or **Excel** format.

