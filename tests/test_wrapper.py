import pytest
import numpy as np
from unittest.mock import Mock, patch

#from zuffy import wrapper
from zuffy.wrapper import ZuffyFitIterator #, ZuffyFitIterator_OLD

@pytest.fixture
def mock_zuffy():
    mock = Mock()
    mock._validate_params = Mock()
    return mock

@pytest.fixture
def sample_data():
    fuzzy_X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([0, 1, 0])
    return fuzzy_X, y

@pytest.fixture
def mock_fit_iterator_results():
    best_est = Mock()
    best_score = 0.85
    iter_perf = [(0.85, 10, {0: 0.8, 1: 0.9})]
    return best_est, best_score, iter_perf

@patch('wrapper.ZuffyFitIterator_OLD')
def test_zuffyfititerator_init_default(mock_iterator_old, mock_zuffy, sample_data):
    fuzzy_X, y = sample_data
    best_est, best_score, iter_perf = mock_fit_iterator_results()
    mock_iterator_old.return_value = (best_est, best_score, iter_perf)
    
    iterator = ZuffyFitIterator(mock_zuffy, fuzzy_X, y)
    
    assert iterator.n_iter == 5
    assert iterator.split_at == 0.2
    assert iterator.random_state == 0
    mock_zuffy._validate_params.assert_called_once()

@patch('wrapper.ZuffyFitIterator_OLD')
def test_zuffyfititerator_custom_params(mock_iterator_old, mock_zuffy, sample_data):
    fuzzy_X, y = sample_data
    best_est, best_score, iter_perf = mock_fit_iterator_results()
    mock_iterator_old.return_value = (best_est, best_score, iter_perf)
    
    iterator = ZuffyFitIterator(mock_zuffy, fuzzy_X, y, n_iter=10, split_at=0.3, random_state=42)
    
    assert iterator.n_iter == 10
    assert iterator.split_at == 0.3
    assert iterator.random_state == 42

def test_get_best_estimator(mock_zuffy, sample_data, mock_fit_iterator_results):
    with patch('wrapper.ZuffyFitIterator_OLD') as mock_iterator_old:
        fuzzy_X, y = sample_data
        best_est, best_score, iter_perf = mock_fit_iterator_results
        mock_iterator_old.return_value = (best_est, best_score, iter_perf)
        
        iterator = ZuffyFitIterator(mock_zuffy, fuzzy_X, y)
        assert iterator.getBestEstimator() == best_est

def test_get_best_score(mock_zuffy, sample_data, mock_fit_iterator_results):
    with patch('wrapper.ZuffyFitIterator_OLD') as mock_iterator_old:
        fuzzy_X, y = sample_data
        best_est, best_score, iter_perf = mock_fit_iterator_results
        mock_iterator_old.return_value = (best_est, best_score, iter_perf)
        
        iterator = ZuffyFitIterator(mock_zuffy, fuzzy_X, y)
        assert iterator.getBestScore() == best_score

def test_invalid_parameters(mock_zuffy, sample_data):
    fuzzy_X, y = sample_data
    
    with pytest.raises(ValueError):
        ZuffyFitIterator(mock_zuffy, fuzzy_X, y, n_iter=0)  # n_iter must be > 0
    
    with pytest.raises(ValueError):
        ZuffyFitIterator(mock_zuffy, fuzzy_X, y, split_at=1.5)  # split_at must be between 0 and 1