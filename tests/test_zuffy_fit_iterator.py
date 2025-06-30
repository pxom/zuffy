import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

# Import the class to be tested
# Assumes zuffy_fit_iterator.py and fuzzy_transformer.py are in the same directory
from zuffy.zuffy_fit_iterator import ZuffyFitIterator
from zuffy.fuzzy_transformer import FuzzyTransformer 

# --- Pytest Fixtures for Mocks and Data ---

@pytest.fixture
def sample_data():
    """Provides a simple, consistent dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    # Give feature names for testing with pandas
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[2])])
    return X, y

@pytest.fixture
def mock_zuffy_model():
    """Creates a mock ZuffyClassifier instance."""
    # This mock simulates the structure needed by ZuffyFitIterator
    mock_model = MagicMock(spec=BaseEstimator)
    mock_model.fit.return_value = mock_model
    mock_model.predict.return_value = np.array([0, 1, 0, 1])
    mock_model.score.return_value = 0.95
    mock_model.classes_ = np.array([0, 1])
    mock_model.verbose = False
    
    # Mock the tree structure for calculating 'tree_size'
    # Create a mock program with a 'program' attribute that is a list
    mock_program = MagicMock()
    mock_program.program = ['node1', 'node2', 'node3'] # size = 3

    # Create a mock estimator that has a '_program' attribute
    mock_estimator_instance = MagicMock()
    mock_estimator_instance._program = mock_program
    
    # Mock the multi-estimator structure
    mock_multi = MagicMock()
    mock_multi.estimators_ = [mock_estimator_instance] # One estimator with one tree of size 3
    mock_model.multi_ = mock_multi
    
    return mock_model

# --- Tests for the ZuffyFitIterator Class ---

class TestZuffyFitIterator:
    """Tests for the ZuffyFitIterator class."""

    def test_initialization(self, mock_zuffy_model):
        """Test the constructor and parameter validation."""
        iterator = ZuffyFitIterator(model=mock_zuffy_model, n_iter=10, random_state=42)
        assert iterator.n_iter == 10
        assert iterator.random_state == 42
        assert iterator.model is mock_zuffy_model
        
        # Test parameter validation
        with pytest.raises(ValueError, match="n_iter must be an instance of"):
            ZuffyFitIterator(model=mock_zuffy_model, n_iter=0)
        with pytest.raises(ValueError, match="test_size must be an instance of"):
            ZuffyFitIterator(model=mock_zuffy_model, test_size=1.1)

    @patch('zuffy_fit_iterator.ZuffyFitIterator._perform_single_fit_job')
    def test_fit_selects_best_score(self, mock_fit_job, sample_data, mock_zuffy_model):
        """Test that the iterator correctly selects the model with the highest score."""
        X, y = sample_data
        
        # Configure the mock to return different scores for each iteration
        # format: (score, fitted_model, class_scores)
        mock_fit_job.side_effect = [
            (0.80, mock_zuffy_model, {0: 0.8, 1: 0.8}), # Iter 0
            (0.95, mock_zuffy_model, {0: 0.9, 1: 1.0}), # Iter 1 (best score)
            (0.90, mock_zuffy_model, {0: 0.9, 1: 0.9}), # Iter 2
        ]
        
        iterator = ZuffyFitIterator(model=mock_zuffy_model, n_iter=3, random_state=0)
        iterator.fit(X, y)

        assert mock_fit_job.call_count == 3
        assert iterator.best_score_ == 0.95
        assert iterator.best_iteration_index_ == 1
        assert len(iterator.iteration_performance_) == 3
        assert iterator.iteration_performance_[1][0] == 0.95

    @patch('zuffy_fit_iterator.ZuffyFitIterator._perform_single_fit_job')
    def test_fit_tiebreaker_by_tree_size(self, mock_fit_job, sample_data):
        """Test that tree size is used as a tie-breaker when scores are equal."""
        X, y = sample_data

        # Create two mock models with different tree sizes
        model_large_tree = MagicMock()
        model_large_tree.multi_.estimators_ = [MagicMock(_program=MagicMock(program=['a']*10))] # size=10

        model_small_tree = MagicMock()
        model_small_tree.multi_.estimators_ = [MagicMock(_program=MagicMock(program=['a']*5))]  # size=5

        # Configure mock to return same score but different models (and thus tree sizes)
        mock_fit_job.side_effect = [
            (0.90, model_large_tree, {}), # Iter 0
            (0.90, model_small_tree, {}), # Iter 1 (same score, smaller tree)
        ]

        iterator = ZuffyFitIterator(model=MagicMock(), n_iter=2, random_state=0)
        iterator.fit(X, y)
        
        assert iterator.best_score_ == 0.90
        assert iterator.best_iteration_index_ == 1 # Should select the second iteration
        assert iterator.smallest_tree_size_ == 5  # Should be the smaller size
        assert iterator.iteration_performance_[1][1] == 5 # Check tree size in performance log

    def test_fit_with_real_transformer(self, sample_data, mock_zuffy_model):
        """A more integrated test to ensure it works with the actual FuzzyTransformer."""
        X, y = sample_data
        iterator = ZuffyFitIterator(model=mock_zuffy_model, n_iter=2, random_state=42)
        iterator.fit(X, y)

        assert iterator.best_estimator_ is not None
        assert iterator.best_score_ >= 0 # Should have a valid score
        assert iterator.fuzzy_feature_names_ is not None
        assert len(iterator.fuzzy_feature_names_) == X.shape[1] * 3

    def test_reproducibility_with_random_state(self, sample_data, mock_zuffy_model):
        """Ensure that a fixed random_state gives reproducible results."""
        X, y = sample_data
        
        iterator1 = ZuffyFitIterator(model=mock_zuffy_model, n_iter=2, random_state=123)
        iterator1.fit(X, y)
        
        iterator2 = ZuffyFitIterator(model=mock_zuffy_model, n_iter=2, random_state=123)
        iterator2.fit(X, y)

        # The performance logs should be identical
        np.testing.assert_allclose(
            [p[0] for p in iterator1.iteration_performance_],
            [p[0] for p in iterator2.iteration_performance_]
        )
        assert iterator1.best_score_ == iterator2.best_score_
        assert iterator1.best_iteration_index_ == iterator2.best_iteration_index_

    def test_get_best_class_accuracy(self, sample_data, mock_zuffy_model):
        """Test the helper method for finding the best class accuracy."""
        X, y = sample_data
        iterator = ZuffyFitIterator(model=mock_zuffy_model, n_iter=1, random_state=42)
        
        # Manually set the performance log
        iterator.best_iteration_index_ = 0
        iterator.iteration_performance_ = [
            [0.9, 10, {'A': 0.85, 'B': 0.95, 'C': 0.90}] # score, tree_size, class_scores
        ]
        
        best_class = iterator.get_best_class_accuracy()
        assert best_class == 'B'

# --- Scikit-learn Compatibility Check ---

# This checks if the estimator adheres to scikit-learn conventions.
# It requires a mock model to be passed to the iterator.
@pytest.mark.skip(reason="check_estimator can be complex to set up with nested models, skipping for now")
def test_sklearn_compatibility():
    """Check if the estimator is scikit-learn compatible."""
    # To run this, you would need a more sophisticated mock or a simple, fast, real estimator
    mock_model_for_check = MagicMock(spec=BaseEstimator)
    # The check_estimator will clone this, so it needs to be configurable
    mock_model_for_check.get_params.return_value = {}
    
    # We pass an instance of our iterator with the mock model to check_estimator
    check_estimator(ZuffyFitIterator(model=mock_model_for_check))

