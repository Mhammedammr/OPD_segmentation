from src.insight import utils
import pandas as pd
import pytest
from src.insight.utils import descriptive_analysis  # Update path as needed

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "gender": ["M", "F", "M", "F", "M"]
    })

def test_standard_dataframe(sample_df):
    num_desc, cat_desc, missing_perc, dups_perc, unq_perc = descriptive_analysis(sample_df)

    assert isinstance(num_desc, pd.DataFrame)
    assert isinstance(cat_desc, pd.DataFrame)
    assert isinstance(missing_perc, pd.DataFrame)
    assert isinstance(dups_perc, float)
    assert isinstance(unq_perc, pd.DataFrame)
