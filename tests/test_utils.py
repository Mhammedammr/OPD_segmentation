import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from insight.utils import (
    descriptive_analysis,
    handle_negative,
    missing_adv,
    imputer,
    IQR,
    IF,
    remove_outliers
)

# ==========================================================================================
# ✅ FIXTURES: Reusable DataFrames for Tests
# ==========================================================================================
@pytest.fixture
def sample_df():
    """Fixture for a sample DataFrame with missing values."""
    return pd.DataFrame({
        "age": [20, None, 40, None, 60],
        "salary": [3000, 6000, None, 12000, None],
        "gender": ["M", None, "M", "F", None],
    })

@pytest.fixture
def sample_outlier_df():
    """Fixture for a sample DataFrame with numerical outliers."""
    return pd.DataFrame({
        "age": [20, 22, 24, 200, 25, 21, 23, 210],  # 200, 210 are outliers
        "salary": [3000, 3200, 3400, 50000, 3600, 3100, 3300, 52000],  # 50000, 52000 are outliers
        "gender": ["M", "F", "M", "M", "F", "F", "M", "F"],
    })


# ✅ TEST CLASS FOR descriptive_analysis()
class TestDescriptiveAnalysis:
    # ✅ Test 1: Standard DataFrame with numerical & categorical columns
    def test_standard_dataframe(self, sample_df):
        num_desc, cat_desc, missing_perc, dups_perc, unq_perc = descriptive_analysis(sample_df)
        assert isinstance(num_desc, pd.DataFrame)
        assert isinstance(cat_desc, pd.DataFrame)
        assert isinstance(missing_perc, pd.DataFrame)
        assert isinstance(dups_perc, float)
        assert isinstance(unq_perc, pd.DataFrame)

    # ✅ Test 2: Empty DataFrame
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        num_desc, cat_desc, missing_perc, dups_perc, unq_perc = descriptive_analysis(df)
        assert num_desc == "No numerical columns in the DataFrame."
        assert cat_desc == "No categorical columns in the DataFrame."
        assert missing_perc.empty
        assert dups_perc == 0
        assert unq_perc.empty

    # ✅ Test 3: DataFrame with Only Numerical Columns
    def test_numerical_dataframe(self):
        df = pd.DataFrame({"age": [20, 25, 30], "salary": [40000, 50000, 60000]})
        num_desc, cat_desc, _, _, _ = descriptive_analysis(df)
        assert isinstance(num_desc, pd.DataFrame)
        assert cat_desc == "No categorical columns in the DataFrame."

    # ✅ Test 4: DataFrame with Only Categorical Columns
    def test_categorical_dataframe(self):
        df = pd.DataFrame({"gender": ["M", "F", "M"], "city": ["NY", "LA", "NY"]})
        num_desc, cat_desc, _, _, _ = descriptive_analysis(df)
        assert num_desc == "No numerical columns in the DataFrame."
        assert isinstance(cat_desc, pd.DataFrame)

    # ✅ Test 5: DataFrame with Missing Values
    def test_missing_values(self):
        df = pd.DataFrame({
            "age": [20, None, 30],
            "salary": [40000, 50000, None]
        })
        _, _, missing_perc, _, _ = descriptive_analysis(df)
        assert missing_perc.loc["age", "missing %"] == 33.33
        assert missing_perc.loc["salary", "missing %"] == 33.33

    # ✅ Test 6: DataFrame with Duplicates
    def test_duplicates(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "age": [25, 30, 25]
        })
        _, _, _, dups_perc, _ = descriptive_analysis(df)
        assert round(dups_perc, 2) == 33.33  # 1 duplicate out of 3 rows

    # ✅ Test 7: Non-DataFrame Input (Should Raise Error)
    def test_invalid_input(self):
        with pytest.raises(AttributeError):  # Since df.select_dtypes() will fail
            descriptive_analysis(["Not a DataFrame"])

# ✅ TEST CLASS FOR handle_negative()
class TestHandleNegative:
    # ✅ Test 1: Remove rows with negative values in keyword columns
    def test_remove_negatives(self):
        df = pd.DataFrame({
            "price": [100, -50, 200, -10],  # "price" should trigger row removal
            "cost": [500, 300, -20, 600],   # "cost" should trigger row removal
            "quantity": [5, 10, 15, 20],    # Not a keyword column (should be ignored)
            "other_col": [-100, 200, 300, -400]  # Should be ignored
        })
        result = handle_negative(df)
        assert len(result) == 1  # Only one row (index 2) should remain
        assert all(result["price"] >= 0)  # No negative values should exist
        assert all(result["cost"] >= 0)

    # ✅ Test 2: No removal when negatives exist in non-keyword columns
    def test_negatives_in_other_columns(self):
        df = pd.DataFrame({
            "quantity": [5, -10, 15, -20],  # Not a keyword column
            "other_col": [-100, 200, 300, -400]  # Not a keyword column
        })
        result = handle_negative(df)
        assert result.equals(df)  # Should remain unchanged

    # ✅ Test 3: No changes when no negatives exist
    def test_no_negatives(self):
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "cost": [500, 600, 700]
        })
        result = handle_negative(df)
        assert result.equals(df)  # Should remain unchanged

    # ✅ Test 4: No keyword columns in DataFrame
    def test_no_keyword_columns(self):
        df = pd.DataFrame({
            "sales": [100, -200, 300],  # Column name does not match keywords
            "units": [5, 10, 15]
        })
        result = handle_negative(df)
        assert result.equals(df)  # Should remain unchanged

    # ✅ Test 5: Empty DataFrame
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = handle_negative(df)
        assert result.empty  # Should return an empty DataFrame

    # ✅ Test 6: Invalid Input (Non-DataFrame)
    def test_invalid_input(self):
        with pytest.raises(AttributeError):  # df.select_dtypes() will fail
            handle_negative(["Not a DataFrame"])




# ✅ FIXTURE: Sample DataFrame with missing values


# ✅ TEST CLASS FOR missing_adv()
class TestMissingAdv:
    # ✅ Test 1: Handle Missing Data by Removing Rows
    def test_remove_missing_data(self, sample_df):
        cfg = {"clean": "Remove Missing Data"}
        result = missing_adv(sample_df, cfg)
        assert result.isna().sum().sum() == 0  # No missing values should remain
        assert len(result) == len(sample_df.dropna())

    # ✅ Test 2: Impute with Mean
    def test_impute_mean(self, sample_df):
        cfg = {"clean": "Impute with Mean"}
        result = missing_adv(sample_df, cfg)
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "age"] == 40  # Mean of [20, 40, 60] = 40
        assert result.loc[2, "salary"] == 9000  # Mean of [3000, 6000, 12000] = 9000

    # ✅ Test 3: Impute with Median
    def test_impute_median(self, sample_df):
        cfg = {"clean": "Impute with Median"}
        result = missing_adv(sample_df, cfg)
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "age"] == 40  # Median of [20, 40, 60] = 40
        assert result.loc[2, "salary"] == 6000  # Median of [3000, 6000, 12000] = 6000

    # ✅ Test 4: Impute with Mode
    def test_impute_mode(self, sample_df):
        cfg = {"clean": "Impute with Mode"}
        result = missing_adv(sample_df, cfg)
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "gender"] == "M"  # Mode of "gender" column

    # ✅ Test 5: Missing "clean" Key in `cfg`
    def test_missing_cfg_key(self, sample_df):
        cfg = {}  # Missing "clean" key
        with pytest.raises(KeyError):
            missing_adv(sample_df, cfg)

    # ✅ Test 6: Invalid Cleaning Method
    def test_invalid_clean_method(self, sample_df):
        cfg = {"clean": "Invalid Method"}
        with pytest.raises(ValueError, match="Invalid cleaning method"):
            missing_adv(sample_df, cfg)

# ✅ TEST CLASS FOR imputer()
class TestImputer:
    # ✅ Test 1: Impute with Mean
    def test_impute_mean(self, sample_df):
        result = imputer(sample_df, strategy="mean")
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "age"] == 40  # Mean of [20, 40, 60] = 40

    # ✅ Test 2: Impute with Median
    def test_impute_median(self, sample_df):
        result = imputer(sample_df, strategy="median")
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "salary"] == 6000  # Median of [3000, 6000, 12000] = 6000

    # ✅ Test 3: Invalid Strategy
    def test_invalid_strategy(self, sample_df):
        with pytest.raises(ValueError, match="Invalid strategy"):
            imputer(sample_df, strategy="invalid_strategy")

    # ✅ Test 4: Empty DataFrame
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot impute an empty DataFrame"):
            imputer(df)


# ✅ TEST CLASS FOR IQR METHOD
class TestIQR:
    # ✅ Test 1: Removing Outliers with IQR
    def test_iqr_removal(self, sample_outlier_df):
        clean_df, outlier_df = IQR(sample_outlier_df, multiplier=1.5)

        # Outliers should be removed
        assert 200 not in clean_df["age"].values
        assert 210 not in clean_df["age"].values
        assert 50000 not in clean_df["salary"].values
        assert 52000 not in clean_df["salary"].values

        # Outlier DataFrame should contain only outliers
        assert all(outlier_df["age"].isin([200, 210]))
        assert all(outlier_df["salary"].isin([50000, 52000]))

    # ✅ Test 2: No Outliers Present
    def test_no_outliers(self):
        df = pd.DataFrame({"age": [20, 22, 24, 25, 21, 23]})
        clean_df, outlier_df = IQR(df, multiplier=1.5)
        assert clean_df.equals(df)  # Should be unchanged
        assert outlier_df.empty  # No outliers



    # ✅ Test 3: Empty DataFrame
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        clean_df, outlier_df = IQR(df, multiplier=1.5)
        assert clean_df.empty
        assert outlier_df.empty

    # ✅ Test 5: Binary Column Ignored
    def test_binary_column_ignored(self):
        df = pd.DataFrame({"binary_col": [0, 1, 1, 0, 1, 0, 1, 1]})
        clean_df, outlier_df = IQR(df, multiplier=1.5)
        assert clean_df.equals(df)  # Binary columns shouldn't be treated as outliers
        assert outlier_df.empty

    # ✅ Test 6: Invalid Multiplier
    def test_invalid_multiplier(self, sample_df):
        with pytest.raises(TypeError):
            IQR(sample_df, multiplier="wrong_type")  # Should raise an error


# ✅ TEST CLASS FOR ISOLATION FOREST METHOD
class TestIF:
    # ✅ Test 1: Removing Outliers with Isolation Forest
    def test_if_removal(self, sample_outlier_df):
        clean_df, outlier_df = IF(sample_outlier_df, contamination=0.25)

        # Ensure outliers are removed
        assert 200 not in clean_df["age"].values
        assert 210 not in clean_df["age"].values
        assert 50000 not in clean_df["salary"].values
        assert 52000 not in clean_df["salary"].values

    

    # ✅ Test 3: Empty DataFrame
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        clean_df, outlier_df = IF(df, contamination=0.25)
        assert clean_df.empty
        assert outlier_df.empty

    # ✅ Test 4: Invalid Contamination
    def test_invalid_contamination(self, sample_df):
        with pytest.raises(ValueError):
            IF(sample_df, contamination=-0.5)  # Should raise an error


# ✅ TEST CLASS FOR remove_outliers()
class TestRemoveOutliers:
    # ✅ Test 1: Using IQR
    def test_remove_outliers_iqr(self, sample_outlier_df):
        config = {"outlier": "Use IQR", "multiplier": 1.5}
        clean_df, outlier_df = remove_outliers(sample_outlier_df, config)

        assert 200 not in clean_df["age"].values
        assert 210 not in clean_df["age"].values
        assert 50000 not in clean_df["salary"].values
        assert 52000 not in clean_df["salary"].values

    # ✅ Test 2: Using Isolation Forest
    def test_remove_outliers_if(self, sample_outlier_df):
        config = {"outlier": "Use Isolation Forest", "contamination": 0.25}
        clean_df, outlier_df = remove_outliers(sample_outlier_df, config)

        assert 200 not in clean_df["age"].values
        assert 210 not in clean_df["age"].values
        assert 50000 not in clean_df["salary"].values
        assert 52000 not in clean_df["salary"].values

    # ✅ Test 3: No Outliers, Should Remain Unchanged
    def test_remove_outliers_no_outliers(self):
        df = pd.DataFrame({"age": [20, 22, 24, 25, 21, 23]})
        config = {"outlier": "Use IQR", "multiplier": 1.5}
        clean_df, outlier_df = remove_outliers(df, config)
        assert clean_df.equals(df)
        assert outlier_df.empty

    # ✅ Test 4: Invalid Config Option
    def test_invalid_config_option(self, sample_outlier_df):
        config = {"outlier": "Unknown Method"}
        with pytest.raises(ValueError):
            remove_outliers(sample_outlier_df, config)