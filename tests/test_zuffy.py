"""
Pytest script for the zuffy.zuffy module, specifically testing the ZuffyClassifier class.

This script tests the functionality of ZuffyClassifier, including its initialization,
fitting process, prediction, and probability prediction, by mocking external
dependencies like gplearn's SymbolicClassifier and scikit-learn's multi-class wrappers.
"""

import pytest
import numpy as np
import numbers # Used for Interval in parameter constraints
import sys
from typing import List, Dict, Optional, Any, Tuple
from unittest.mock import MagicMock, create_autospec

# --- Mocking specific sklearn utility functions as they are used directly ---
# These are internal utility functions that ZuffyClassifier calls.
# We'll replace them with simple mock implementations for testing.

# Minimal mock for check_is_fitted
def check_is_fitted_mock(estimator):
    # This mock ensures that `hasattr(estimator, 'multi_')` is True and `estimator.multi_` is not None.
    # In actual sklearn, it checks for attributes ending with an underscore.
    if not hasattr(estimator, 'multi_') or estimator.multi_ is None:
        raise RuntimeError("Estimator not fitted.")

# Minimal mock for check_classification_targets
def check_classification_targets_mock(y):
    if len(np.unique(y)) < 2:
        raise ValueError("y contains less than 2 unique classes for classification.")


# --- Pytest Fixtures ---

@pytest.fixture
def dummy_data():
    """Returns a simple dummy dataset for classification."""
    X = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5], [1.2, 2.2], [5.0, 6.0], [5.5, 6.5]])
    y = np.array([0, 0, 0, 0, 1, 1])
    return X, y

@pytest.fixture(autouse=True)
def mock_gplearn_symbolic_classifier(mocker):
    """
    Patches gplearn.genetic.SymbolicClassifier to return a MagicMock instance.
    This mock instance explicitly defines its 'fit', 'predict', 'predict_proba' methods.
    It also handles potential import order issues by clearing sys.modules.
    """
    # Create a MagicMock for the SymbolicClassifier instance
    mock_sc_instance = mocker.MagicMock() 
    
    # Explicitly set return values and ensure methods are callable mocks
    mock_sc_instance.fit.return_value = mock_sc_instance
    mock_sc_instance.predict.return_value = np.array([0, 1]) # Default prediction for 2 classes
    mock_sc_instance.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]]) # Default probabilities

    # IMPORTANT: Clear zuffy.zuffy and gplearn.genetic from sys.modules
    # to ensure that when ZuffyClassifier is imported later, it picks up this patch.
    if 'zuffy.zuffy' in sys.modules:
        del sys.modules['zuffy.zuffy']
    if 'gplearn.genetic' in sys.modules:
        del sys.modules['gplearn.genetic']

    # Patch the class itself in its original module path to return our custom mock instance
    mocker.patch('gplearn.genetic.SymbolicClassifier', return_value=mock_sc_instance)
    return mock_sc_instance # Return the mock instance for direct assertion in tests

@pytest.fixture(autouse=True)
def mock_sklearn_multiclass_wrappers(mocker):
    """
    Patches sklearn.multiclass.OneVsRestClassifier and OneVsOneClassifier classes
    to return MagicMock instances that mimic their behavior, particularly
    accepting an 'estimator' argument and ensuring 'fit', 'predict', 'predict_proba' exist.
    """
    
    # Mock constructor for OneVsRestClassifier
    def mock_ovr_constructor_side_effect(*args, **kwargs):
        # Create a MagicMock instance
        instance = mocker.MagicMock()
        # Explicitly set return values for methods, ensuring they exist
        instance.fit.return_value = instance
        instance.predict.return_value = np.array([0, 0, 1, 1])
        instance.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        
        # Store the estimator passed during construction
        instance.estimator = kwargs.get('estimator', args[0] if args else None)
        return instance

    # Patch the class in its original module path
    mocker.patch('sklearn.multiclass.OneVsRestClassifier', side_effect=mock_ovr_constructor_side_effect)

    # Mock constructor for OneVsOneClassifier
    def mock_ovo_constructor_side_effect(*args, **kwargs):
        # Create a MagicMock instance
        instance = mocker.MagicMock()
        # Explicitly set return values for methods, ensuring they exist
        instance.fit.return_value = instance
        instance.predict.return_value = np.array([0, 0, 1, 1])
        instance.predict_proba.return_value = np.array([[0.95, 0.05], [0.85, 0.15], [0.15, 0.85], [0.05, 0.95]])
        
        # Store the estimator passed during construction
        instance.estimator = kwargs.get('estimator', args[0] if args else None)
        return instance

    # Patch the class in its original module path
    mocker.patch('sklearn.multiclass.OneVsOneClassifier', side_effect=mock_ovo_constructor_side_effect)


@pytest.fixture(autouse=True)
def mock_sklearn_utils_and_base_estimator(mocker):
    """
    Mocks external sklearn utility functions and BaseEstimator._validate_data
    that ZuffyClassifier imports and uses.
    """
    # Patch check_is_fitted in its original module path
    mocker.patch('sklearn.utils.validation.check_is_fitted', new=check_is_fitted_mock)

    # Patch check_classification_targets in its original module path
    mocker.patch('sklearn.utils.multiclass.check_classification_targets', new=check_classification_targets_mock)

    # Patch sklearn.base.BaseEstimator._validate_data
    from sklearn.base import BaseEstimator as SklearnBaseEstimator

    # Capture the original _validate_data method
    # This is not strictly necessary for the fix but good practice if needed for fallback
    original_validate_data = SklearnBaseEstimator._validate_data
    
    def _mock_validate_data_side_effect(self_estimator, X_input, y_input=None, reset=True, **kwargs):
        # Simulate the behavior of sklearn's _validate_data for testing purposes.
        # This includes setting n_features_in_ and performing checks.
        
        if reset:
            self_estimator.n_features_in_ = X_input.shape[1]
            # Mock feature_names_in_ if X is pandas-like, similar to sklearn's internal logic
            if hasattr(X_input, 'columns'):
                self_estimator.feature_names_in_ = np.array(X_input.columns, dtype=object)
            else:
                self_estimator.feature_names_in_ = None
        elif hasattr(self_estimator, 'n_features_in_') and X_input.shape[1] != self_estimator.n_features_in_:
            # Simulate feature mismatch check for predict/predict_proba when reset=False
            raise ValueError(
                f"X has {X_input.shape[1]} features, but this estimator was fitted with "
                f"{self_estimator.n_features_in_} features."
            )
        # For other valid calls, simply return X_input and y_input as they are validated
        return X_input, y_input

    # Patch the method on the actual sklearn.base.BaseEstimator class
    mocker.patch('sklearn.base.BaseEstimator._validate_data', new=_mock_validate_data_side_effect)


# --- Test Cases for ZuffyClassifier ---

class TestZuffyClassifier:

    # Import ZuffyClassifier inside the test class methods
    # to ensure it's imported *after* the mocking fixtures have run
    @pytest.fixture(scope='class', autouse=True)
    def import_zuffy_classifier(self):
        # Import ZuffyClassifier once for all tests in this class
        # This ensures the mocks are in place when ZuffyClassifier is loaded
        global ZuffyClassifier # To assign to the global ZuffyClassifier
        from zuffy.zuffy import ZuffyClassifier as _ZuffyClassifier
        ZuffyClassifier = _ZuffyClassifier

    def test_init_default_parameters(self):
        """Test ZuffyClassifier initializes with correct default parameters."""
        clf = ZuffyClassifier()
        assert clf.generations == 20
        assert clf.population_size == 1000
        assert clf.multiclassifier == 'OneVsRestClassifier'
        assert clf.function_set == ZuffyClassifier.default_function_set
        assert clf.verbose == 0
        assert clf.random_state is None
        assert clf.parsimony_coefficient == 0.001

    def test_init_custom_parameters(self):
        """Test ZuffyClassifier initializes with custom parameters."""
        # Import operators inside the test function scope if needed
        # to ensure they are available after ZuffyClassifier import
        from zuffy._fpt_operators import COMPLEMENT
        custom_function_set = [COMPLEMENT]
        clf = ZuffyClassifier(
            generations=5,
            population_size=50,
            multiclassifier='OneVsOneClassifier',
            function_set=custom_function_set,
            verbose=1,
            random_state=42,
            parsimony_coefficient="auto"
        )
        assert clf.generations == 5
        assert clf.population_size == 50
        assert clf.multiclassifier == 'OneVsOneClassifier'
        assert clf.function_set == custom_function_set
        assert clf.verbose == 1
        assert clf.random_state == 42
        assert clf.parsimony_coefficient == "auto"

    @pytest.mark.parametrize("invalid_multiclassifier", ["Invalid", "OVRO", 123])
    def test_fit_invalid_multiclassifier_raises_error(self, dummy_data, invalid_multiclassifier):
        """Test that fit raises ValueError for invalid multiclassifier option."""
        X, y = dummy_data
        clf = ZuffyClassifier(multiclassifier=invalid_multiclassifier)
        with pytest.raises(ValueError, match="multiclassifier must be one of: OneVsOneClassifier, OneVsRestClassifier."):
            clf.fit(X, y)

    def test_fit_stores_data_and_classes(self, dummy_data):
        """Test that fit stores X_, y_ and classes_ correctly."""
        X, y = dummy_data
        clf = ZuffyClassifier()
        
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.X_, X)
        np.testing.assert_array_equal(clf.y_, y)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
        assert hasattr(clf, 'n_features_in_')
        assert clf.n_features_in_ == X.shape[1]

    def test_fit_calls_onevsrestclassifier_fit(self, dummy_data, mocker):
        """Test that ZuffyClassifier uses OneVsRestClassifier and calls its fit method."""
        X, y = dummy_data
        clf = ZuffyClassifier(multiclassifier='OneVsRestClassifier')
        clf.fit(X, y)
        
        # Access the specific mock instance that was created by the side_effect for OVR
        mock_ovr_class = mocker.patch('sklearn.multiclass.OneVsRestClassifier')
        mock_ovr_instance = mock_ovr_class.call_args.return_value
        
        mock_ovr_instance.fit.assert_called_once_with(X, y)
        
        # Verify that the SymbolicClassifier mock was passed as estimator
        mock_sc_class = mocker.patch('gplearn.genetic.SymbolicClassifier')
        assert mock_ovr_instance.estimator is mock_sc_class.call_args.return_value


    def test_fit_calls_onevsoneclassifier_fit(self, dummy_data, mocker):
        """Test that ZuffyClassifier uses OneVsOneClassifier and calls its fit method."""
        X, y = dummy_data
        clf = ZuffyClassifier(multiclassifier='OneVsOneClassifier')
        clf.fit(X, y)

        # Access the specific mock instance created by the side_effect for OVO
        mock_ovo_class = mocker.patch('sklearn.multiclass.OneVsOneClassifier')
        mock_ovo_instance = mock_ovo_class.call_args.return_value

        mock_ovo_instance.fit.assert_called_once_with(X, y)
        
        # Verify that the SymbolicClassifier mock was passed as estimator
        mock_sc_class = mocker.patch('gplearn.genetic.SymbolicClassifier')
        assert mock_ovo_instance.estimator is mock_sc_class.call_args.return_value


    def test_fit_non_classification_targets_raises_error(self, dummy_data, mocker):
        """Test that fit raises ValueError for non-classification targets."""
        X, y = dummy_data
        # Create a y with only one unique class
        y_single_class = np.array([0, 0, 0, 0, 0, 0])
        clf = ZuffyClassifier()
        # Mock the check_classification_targets to raise error when expected
        mocker.patch('sklearn.utils.multiclass.check_classification_targets', side_effect=ValueError("y contains less than 2 unique classes for classification."))
        with pytest.raises(ValueError, match="y contains less than 2 unique classes for classification."):
            clf.fit(X, y_single_class)

    def test_predict_not_fitted_raises_error(self, dummy_data):
        """Test that predict raises RuntimeError if not fitted."""
        X, _ = dummy_data
        clf = ZuffyClassifier()
        with pytest.raises(RuntimeError, match="Estimator not fitted."):
            clf.predict(X)

    def test_predict_success(self, dummy_data, mocker):
        """Test that predict returns correct predictions after fitting."""
        X, y = dummy_data
        clf = ZuffyClassifier()
        clf.fit(X, y)

        # Get the actual mock instance that was created by the side_effect for OVR
        mock_wrapper_instance = mocker.patch('sklearn.multiclass.OneVsRestClassifier').call_args.return_value
        
        X_predict = np.array([[0.1, 0.1], [1.1, 1.1]])
        # Set return value on the specific mock instance that predict will call
        mock_wrapper_instance.predict.return_value = np.array([0, 0]) 

        predictions = clf.predict(X_predict)
        mock_wrapper_instance.predict.assert_called_once_with(X_predict)
        np.testing.assert_array_equal(predictions, np.array([0, 0]))
        assert predictions.shape == (X_predict.shape[0],)

    def test_predict_proba_not_fitted_raises_error(self, dummy_data):
        """Test that predict_proba raises RuntimeError if not fitted."""
        X, _ = dummy_data
        clf = ZuffyClassifier()
        with pytest.raises(RuntimeError, match="Estimator not fitted."):
            clf.predict_proba(X)

    def test_predict_proba_success(self, dummy_data, mocker):
        """Test that predict_proba returns correct probabilities after fitting."""
        X, y = dummy_data
        clf = ZuffyClassifier()
        clf.fit(X, y)

        # Get the actual mock instance that was created by the side_effect for OVR
        mock_wrapper_instance = mocker.patch('sklearn.multiclass.OneVsRestClassifier').call_args.return_value

        X_predict = np.array([[0.1, 0.1], [1.1, 1.1]])
        # Set return value on the specific mock instance that predict_proba will call
        mock_wrapper_instance.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]]) 

        probabilities = clf.predict_proba(X_predict)
        mock_wrapper_instance.predict_proba.assert_called_once_with(X_predict)
        np.testing.assert_array_equal(probabilities, np.array([[0.9, 0.1], [0.2, 0.8]]))
        assert probabilities.shape == (X_predict.shape[0], len(clf.classes_))

    def test_predict_proba_feature_mismatch(self, dummy_data, mocker):
        """Test predict_proba raises ValueError for feature mismatch (handled by _validate_data)."""
        X_train, y_train = dummy_data
        clf = ZuffyClassifier()
        clf.fit(X_train, y_train)

        # Create X_test with a different number of features
        X_test_mismatch = np.array([[1.0, 2.0, 3.0]]) # 3 features instead of 2

        with pytest.raises(ValueError, match="X has 3 features, but this estimator was fitted with 2 features."):
            clf.predict_proba(X_test_mismatch)
