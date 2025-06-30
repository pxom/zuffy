import pytest
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Import the functions and class from your module
# Assuming your code is in 'fpt.py'
import zuffy.fuzzy_transformer as fpt

# --- Pytest Fixtures for Reusable Test Data ---

@pytest.fixture
def sample_dataframe():
    """Provides a standard DataFrame for testing."""
    return pd.DataFrame({
        'temp': [10, 20, 25, 30, 40],
        'humidity': [60, 65, 70, 75, 80],
        'wind': ['low', 'low', 'med', 'high', 'high'],
        #'mood': ['happy', 'happy', 'grand', 'sad', 'sad'],
    })

@pytest.fixture
def sample_numpy_array():
    """Provides a NumPy array equivalent to the numeric part of sample_dataframe."""
    return np.array([
        [10.0, 60.0],
        [20.0, 65.0],
        [25.0, 70.0],
        [30.0, 75.0],
        [40.0, 80.0]
    ])

# --- Tests for the `trimf` function ---

class TestTrimf:
    """Tests for the triangular membership function."""
    
    def test_basic_functionality(self):
        """Test standard behavior of the trimf function."""
        feature = np.array([5, 10, 15, 20, 25, 30, 35])
        abc = [10, 20, 30]
        expected = np.array([0, 0, 0.5, 1, 0.5, 0, 0])
        np.testing.assert_allclose(fpt.trimf(feature, abc), expected)

    def test_edge_cases(self):
        """Test values at the exact a, b, and c points."""
        feature = np.array([10, 20, 30])
        abc = [10, 20, 30]
        expected = np.array([0, 1, 0])
        np.testing.assert_allclose(fpt.trimf(feature, abc), expected)

    def test_degenerate_triangles(self):
        """Test cases where a=b or b=c."""
        feature = np.array([10, 15, 20])
        # Case 1: a = b (left-sided triangle)
        abc1 = [10, 10, 20]
        expected1 = np.array([1.0, 0.5, 0.0])
        np.testing.assert_allclose(fpt.trimf(feature, abc1), expected1)
        
        # Case 2: b = c (right-sided triangle)
        abc2 = [10, 20, 20]
        expected2 = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(fpt.trimf(feature, abc2), expected2)

    def test_invalid_abc_parameters(self):
        """Test that incorrect `abc` parameters raise ValueErrors."""
        feature = np.array([1, 2, 3])
        # Incorrect length
        with pytest.raises(ValueError, match="must have exactly three elements"):
            fpt.trimf(feature, [10, 20])
        # Incorrect order
        with pytest.raises(ValueError, match="must satisfy the condition `a <= b <= c`"):
            fpt.trimf(feature, [20, 10, 30])

# --- Tests for the `convert_to_numeric` function ---

class TestConvertToNumeric:
    """Tests for the label encoding utility function."""

    def test_conversion(self, sample_dataframe):
        """Test the basic conversion of a categorical column."""
        df = sample_dataframe.copy()
        classes, new_df = fpt.convert_to_numeric(df, 'wind')
        
        # Check that original classes are correctly captured
        np.testing.assert_array_equal(classes, ['high', 'low', 'med'])
        
        # Check that the column is converted to integers
        expected_col = np.array([1, 1, 2, 0, 0])
        np.testing.assert_array_equal(new_df['wind'].values, expected_col)
        assert pd.api.types.is_integer_dtype(new_df['wind'])

# --- Tests for the `FuzzyTransformer` class ---

class TestFuzzyTransformer:
    """Comprehensive tests for the FuzzyTransformer."""

    def test_init(self):
        """Test the transformer's initialization."""
        # Test default initialization
        ft = fpt.FuzzyTransformer()
        assert ft.non_fuzzy == []
        assert not ft.oob_check
        assert ft.tags == ['low', 'med', 'high']
        
        # Test custom initialization
        ft_custom = fpt.FuzzyTransformer(
            non_fuzzy=['wind'], 
            oob_check=True, 
            tags=['cold', 'warm', 'hot']
        )
        assert ft_custom.non_fuzzy == ['wind']
        assert ft_custom.oob_check
        assert ft_custom.tags == ['cold', 'warm', 'hot']

        # Test invalid tags
        with pytest.raises(ValueError, match="`tags` must be a list or tuple of three strings."):
            fpt.FuzzyTransformer(tags=['a', 'b'])

    def test_fit_dataframe(self, sample_dataframe):
        """Test fitting the transformer with a pandas DataFrame."""
        ft = fpt.FuzzyTransformer(non_fuzzy=['wind'])
        ft.fit(sample_dataframe)
        
        # Check that attributes are correctly set
        check_is_fitted(ft)
        pd.testing.assert_index_equal(ft.columns_, sample_dataframe.columns)
        
        # Check fuzzy bounds
        assert 'temp' in ft.fuzzy_bounds_
        assert 'humidity' in ft.fuzzy_bounds_
        np.testing.assert_allclose(ft.fuzzy_bounds_['temp'], (10, 25, 40))
        
        # Check categorical values
        assert 'wind' in ft.categorical_values_
        assert ft.categorical_values_['wind'] == ['high', 'low', 'med']
        
        # Check output feature names
        assert len(ft.feature_names_out_) == (2 * 3) + 3 # 2 fuzzy, 1 categorical

    def test_fit_numpy(self, sample_numpy_array):
        """Test fitting with a NumPy array."""
        ft = fpt.FuzzyTransformer(feature_names=['temp', 'humidity'])
        ft.fit(sample_numpy_array)
        
        check_is_fitted(ft)
        assert len(ft.feature_names_out_) == 2 * 3
        np.testing.assert_allclose(ft.fuzzy_bounds_['temp'], (10, 25, 40))

    def test_transform_dataframe(self, sample_dataframe):
        """Test transforming a DataFrame."""
        #X = sample_dataframe.drop(['mood'],axis=1)
        #y = sample_dataframe['mood']
        ft = fpt.FuzzyTransformer(non_fuzzy=['wind'])
        ft.fit(sample_dataframe)
        transformed = ft.transform(sample_dataframe)
        
        # Check output shape
        assert transformed.shape == (5, 3+3+3) # 2 fuzzified and one hot
        
        # Check a fuzzy value (temp=25, which is the midpoint 'b')
        # temp_low, temp_med, temp_high for row 2
        expected_temp_fuzzy = [fpt.trimf(25, [10,10,25]), fpt.trimf(25, [10,25,40]), fpt.trimf(25, [25,40,40])]
        np.testing.assert_allclose(transformed[2, 0:3], expected_temp_fuzzy) # Should be [0, 1, 0]
        
        # Check a categorical value (wind='med' for row 2)
        # wind=high, wind=low, wind=med
        expected_wind_onehot = [0, 0, 1]
        np.testing.assert_array_equal(transformed[2, 6:9], expected_wind_onehot)

    def test_oob_logic(self, sample_dataframe):
        """Test the oob checking functionality."""
        X_test = pd.DataFrame({'temp': [5, 45], 'humidity': [50, 90], 'wind': ['low', 'high']})

        # With oob_check=True (default), it should raise an error
        ft_oob_check = fpt.FuzzyTransformer(non_fuzzy=['wind'], oob_check=True)
        ft_oob_check.fit(sample_dataframe)
        with pytest.raises(ValueError, match="less than 'a'"):
            ft_oob_check.transform(X_test)

        # With oob_check=False, it should ignore values outside the range
        ft_no_oob_check = fpt.FuzzyTransformer(non_fuzzy=['wind'], oob_check=False)
        ft_no_oob_check.fit(sample_dataframe)
        transformed = ft_no_oob_check.transform(X_test)
        
        # The out-of-bounds value (temp=5) should result in [0, 0, 0] fuzzy values
        # temp_low, temp_med, temp_high because they have zero membership of those sets
        np.testing.assert_allclose(transformed[0, 0:3], [0, 0, 0])
        # The other out-of-bounds value (temp=45) should result in [0, 0, 1]
        np.testing.assert_allclose(transformed[1, 0:3], [0, 0, 0])

    def test_not_fitted_error(self, sample_dataframe):
        """Test that transform raises an error if not fitted."""
        ft = fpt.FuzzyTransformer()
        with pytest.raises(NotFittedError):
            ft.transform(sample_dataframe)

    def test_mismatched_columns_error(self, sample_dataframe):
        """Test ValueError for mismatched columns during transform."""
        ft = fpt.FuzzyTransformer(non_fuzzy=['wind'])
        ft.fit(sample_dataframe)
        
        X_wrong_cols = sample_dataframe.rename(columns={'temp': 'temperature'})
        with pytest.raises(ValueError, match="feature names should match those that were passed during fit"):
            ft.transform(X_wrong_cols)

    def test_missing_category_in_transform(self, sample_dataframe):
        """Test handling of data where a category is missing during transform."""
        ft = fpt.FuzzyTransformer(non_fuzzy=['wind'])
        ft.fit(sample_dataframe) # Fitted on ['low', 'med', 'high']

        # Create a test set without 'med' wind
        X_test = sample_dataframe[sample_dataframe['wind'] != 'med'].copy()
        
        transformed = ft.transform(X_test)
        
        # The output shape must be consistent with the fit stage (9 columns)
        assert transformed.shape == (len(X_test), 9)
        
        # The one-hot encoding for 'wind' must still have 3 columns,
        # with the 'med' column being all zeros.
        # columns are high, low, med
        wind_cols = transformed[:, 6:9]
        med_col = wind_cols[:, 2]
        assert np.all(med_col == 0)


# --- Scikit-learn Compatibility Check ---

# This check runs a suite of tests to ensure the transformer adheres to sklearn conventions.
# Note: It requires `check_estimator` which might have specific data expectations.
# We pass a simple instance of the transformer.
@pytest.mark.parametrize(
    "estimator",
    [fpt.FuzzyTransformer(feature_names=['a', 'b']), # ,   'c','d','e'
     fpt.FuzzyTransformer(non_fuzzy=['c'], feature_names=['a','b','c'])] # ,   'd','e'
)
def test_sklearn_compatibility(estimator):
    """Check if the transformer is scikit-learn compatible."""
    check_estimator(estimator)

