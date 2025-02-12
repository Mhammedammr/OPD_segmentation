import pytest
import pandas as pd
import numpy as np
from insight.utils import (
    descriptive_analysis,
    handle_negative,
    missing_adv,
   imputer
)

# ✅ FIXTURE: Reusable sample DataFrame for tests
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, None, 40, None, 60],  # Numeric column with missing values
        "salary": [3000, 6000, None, 12000, None],  # Numeric column with missing values
        "gender": ["M", None, "M", "F", None],  # Categorical column with missing values
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

