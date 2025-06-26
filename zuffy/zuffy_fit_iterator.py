"""
This module contains the ZuffyFitIterator, a scikit-learn compatible
meta-estimator that repeatedly fits a classifier and builds Fuzzy 
Pattern Trees to find an optimal model.

"""

import numbers # for scikit learn Interval
import time
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import Interval, validate_params
from zuffy.fuzzy_transformer import FuzzyTransformer


class ZuffyFitIterator(BaseEstimator):
    """
    Iteratively trains a Zuffy classifier on fuzzified data and tracks performance metrics.

    This class runs multiple randomised train/test splits, fuzzifies the data
    within each split, and then trains a Zuffy classifier (or a GridSearchCV
    wrapper) to find the best-performing model across iterations. It considers
    both accuracy and model complexity (tree size) for model selection.

    Parameters
    ----------
    model : estimator object
        A Zuffy classifier that follows scikit-learn's API. This can also be
        a `GridSearchCV` object wrapping a Zuffy classifier.

    tags : list of str, default=['lo', 'med', 'hi']
        A list of string tags to use for fuzzification, representing the names
        of the fuzzy sets (e.g., low, medium, high). These are passed directly
        to the `FuzzyTransformer`.

    n_iter : int, default=5
        The number of random splits and evaluations to perform. A higher value
        increases the robustness of the results but also increases computation time.

    test_size : float, default=0.2
        The proportion of the dataset to include in the test split for each
        iteration. Must be between 0.0 and 1.0.

    random_state : int or None, default=None
        Controls the randomness of the train/test splits.
        - Pass an `int` for reproducible output across multiple function calls.
        - Pass `None` (default) for a different random state each time.

    Attributes
    ----------
    best_estimator\_ : object
        The best trained model (estimator) found across all iterations.
        This will be a Zuffy classifier or the `best_estimator_` from
        `GridSearchCV` if used.

    best_score\_ : float
        The overall score achieved by the `best_estimator_` on its respective
        test set (accuracy).

    iteration_performance\_ : list of tuples
        A list containing the performance metrics for each iteration.
        Each tuple contains:

        - `score` (float): The overall accuracy of the model on the test set.
        - `tree_size` (int): The total number of nodes in the Zuffy tree(s).
        - `class_scores_dict` (dict): A dictionary mapping class labels to
          their individual accuracy scores for that iteration.

    best_iteration_index\_ : int
        The index of the iteration (0-based) that yielded the `best_estimator_`.

    smallest_tree_size\_ : int
        The size (total number of nodes) of the tree in the `best_estimator_`.
        If `best_estimator_` is a `GridSearchCV`, this refers to the size of
        the best model found within `GridSearchCV`.

    fuzzy_feature_names\_ : list of str
        The names of the features after fuzzification, derived from the
        `FuzzyTransformer` associated with the `best_estimator_`. This
        attribute is set during the `fit` method.

    feature_names_in\_ : ndarray of str
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings (e.g., a Pandas DataFrame).

    n_features_in\_ : int
        The number of features seen during `fit`.
    """

    # Using _parameter_constraints as per scikit-learn convention for validation
    _parameter_constraints: dict = {
        "tags": [list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "test_size": [Interval(numbers.Real, 0, 1, closed="both")],
        "random_state": ["random_state"],
    }

    @validate_params(
        _parameter_constraints,
        prefer_skip_nested_validation=True,
    )
    def __init__(self, model, tags: List[str] = ['lo', 'med', 'hi'], n_iter: int = 5,
                 test_size: float = 0.2, random_state: Union[int, None] = None):

        self.model = model
        if hasattr(self.model, "_validate_params"):
            self.model._validate_params()

        self.tags = tags
        self.n_iter = n_iter
        self.test_size = test_size
        self.random_state = random_state

        # Initialize attributes that will be populated after fitting
        #self.best_estimator_ = None
        #self.best_score_ = -np.inf
        #self.iteration_performance_ = []
        #self.best_iteration_index_ = -1
        #self.smallest_tree_size_ = np.inf
        #self.fuzzy_feature_names_ = None

    @validate_params( 
        {
            "X": ["array-like"],
            "y": ["array-like"]
        }, 
        prefer_skip_nested_validation=True
    )
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None, non_fuzzy=None) -> "ZuffyFitIterator":
        """
        Fits the ZuffyFitIterator by running multiple training iterations.

        This method orchestrates the iterative training process:
        1. Validates input data `X` and `y`.
        2. Performs `n_iter` randomised train/test splits.
        3. For each split, fuzzifies the data using `FuzzyTransformer`.
        4. Trains the `model` (or `GridSearchCV`) on the fuzzified training data.
        5. Evaluates the trained model on the fuzzified test data.
        6. Tracks performance metrics (score, tree size, per-class accuracies).
        7. Selects the `best_estimator_` based on overall score, then smallest tree size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features (unfuzzified) to be used for training and testing.

        y : array-like of shape (n_samples,)
            The target labels corresponding to `X`.

        Returns
        -------
        self : ZuffyFitIterator
            The fitted instance of the ZuffyFitIterator.

        """

        # Preserve feature names if we get a DataFrame rather than a Numpy array
        if isinstance(X, pd.DataFrame) and feature_names is None:
            feature_names = X.columns

        if feature_names is None:
            # generate feature_names
            feature_names = [f'X{i}' for i in range(X.shape[1])]

        #X, y = self._validate_data(X, y, accept_sparse=False, force_all_finite='allow-nan')
        num_X = X.drop(non_fuzzy,axis=1)
        if num_X.shape[1] > 0:
            temp_X, y = self._validate_data(num_X, y, accept_sparse=False, force_all_finite='allow-nan')
        else:
            print("Warning: There are no numeric columns in this dataset")
        self.feature_names_in_ = feature_names
        self.non_fuzzy = non_fuzzy

        best_score_overall = -np.inf
        best_iter_idx = -1
        smallest_tree_size_overall = np.inf
        iteration_performance_list = []
        sum_scores = 0.0

        for i in range(self.n_iter):
            iter_start_time = time.time()
            # Increment random_state for each iteration to get different, but reproducible, splits.
            current_random_state = self.random_state + i if self.random_state is not None else None

            # Perform a single fit job, which includes fuzzification and model training.
            score, current_estimator, class_scores, fuzz_transformer = self._perform_single_fit_job(
                model=self.model,
                X=X,
                y=y,
                test_size=self.test_size,
                random_state=current_random_state,
            )
            sum_scores += score
            self._verbose_out(f"Class scores for iteration {i}: {class_scores}")

            # Determine the actual Zuffy estimator to calculate its tree size.
            # This handles both direct Zuffy models and GridSearchCV results.
            if isinstance(current_estimator, GridSearchCV):
                # Ensure the best_estimator_ exists if GridSearchCV didn't find a valid model
                if not hasattr(current_estimator, 'best_estimator_'):
                    self._verbose_out(f"Warning: GridSearchCV in iteration {i} did not find a best estimator.")
                else:
                    zuffy_estimator = current_estimator.best_estimator_
            else:
                zuffy_estimator = current_estimator

            # Calculate the size of the model (assuming Zuffy models have a 'multi_' attribute)
            tree_size = 0
            if hasattr(zuffy_estimator, 'multi_') and hasattr(zuffy_estimator.multi_, 'estimators_'):
                for e in zuffy_estimator.multi_.estimators_:
                    if hasattr(e, '_program'):
                        tree_size += len(e._program.program)
            self._verbose_out(f"Tree size for iteration {i}: {tree_size}")

            iteration_performance_list.append([score, tree_size, class_scores])

            # Update the best estimator based on score (primary) and then tree size (tie-breaker).
            if (score > best_score_overall) or \
               ((score == best_score_overall) and (tree_size < smallest_tree_size_overall)):
                best_iter_idx = i
                self.best_estimator_ = current_estimator
                self.fuzz_transformer_ = fuzz_transformer
                best_score_overall = score
                smallest_tree_size_overall = tree_size
                self._verbose_out(f"\aNew best estimator found: Iteration {i} with score {score:.5f} and tree size {tree_size}")

            iter_duration = round(time.time() - iter_start_time, 1)
            avg_score_so_far = sum_scores / (i + 1)
            self._verbose_out(f"Iteration #{i} took {iter_duration}s | Best so far: {best_score_overall:.5f} (size: {smallest_tree_size_overall}) | Average score: {avg_score_so_far:.5f}")

        self._verbose_out(f"Finished iterating. Best iteration index: {best_iter_idx}")
        self.best_score_ = best_score_overall
        self.iteration_performance_ = iteration_performance_list
        self.best_iteration_index_ = best_iter_idx
        self.smallest_tree_size_ = smallest_tree_size_overall
        return self

    def _perform_single_fit_job(self, model: Any, X: np.ndarray, y: np.ndarray,
                                test_size: float = 0.2, random_state: Union[int, None] = None) \
                                -> Tuple[float, Any, Dict[Any, float], List[str]]:
        """
        Performs a single iteration of data splitting, fuzzification, model training,
        and evaluation.

        This private method handles one randomised train/test split, fuzzifies the
        resulting data, trains the provided model, and calculates performance metrics.

        Parameters
        ----------
        model : estimator object
            A Zuffy classifier or `GridSearchCV` object to be trained.

        X : array-like of shape (n_samples, n_features)
            The input features for this single job.

        y : array-like of shape (n_samples,)
            The target labels for this single job.

        test_size : float, default=0.2
            The proportion of the dataset to include in the test split.

        random_state : int or None, default=None
            The random seed for the `train_test_split`.

        Returns
        -------
        score : float
            The overall accuracy score of the fitted model on the test set.

        fitted_model : object
            The trained model (either the base Zuffy classifier or the
            `best_estimator_` from `GridSearchCV`).

        class_scores : dict
            A dictionary where keys are class labels and values are their
            corresponding accuracy scores on the test set for this iteration.

        fuzzy_feature_names : list of str
            A list of feature names generated by the `FuzzyTransformer` for
            this specific iteration.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y, random_state=random_state
        )
        
        fuzz_transformer = FuzzyTransformer(feature_names=self.feature_names_in_, non_fuzzy=self.non_fuzzy, tags=self.tags)
        fuzz_transformer.fit(X_train)
        fuzzy_X_train = fuzz_transformer.transform(X_train)
        self.fuzzy_feature_names_ = fuzz_transformer.feature_names_out_

        fitted_model = model.fit(fuzzy_X_train, y_train) ### ??? n jobs here?
        fuzzy_X_test = fuzz_transformer.transform(X_test)
        score = fitted_model.score(fuzzy_X_test, y_test)
        self._verbose_out(f"Overall test score: {score:.8f}")

        predictions = fitted_model.predict(fuzzy_X_test)
        class_scores = {}
        # Calculate individual class accuracies.
        for cls in np.unique(fitted_model.classes_):
            idx = (y_test == cls)
            if np.any(idx):  # Ensure there are samples for the current class in the test split
                class_accuracy = accuracy_score(y_test[idx], predictions[idx])
                class_scores[cls] = class_accuracy
                self._verbose_out(f"Accuracy for class {cls}: {class_accuracy:.5f}")
            else:
                self._verbose_out(f"Class {cls} not present in this test split.")

        avg_score = round(np.mean(list(class_scores.values())), 5)
        self._verbose_out(f"Average Class score: {avg_score} [DIFF={(score-avg_score):.5f}]")

        return score, fitted_model, class_scores, fuzz_transformer
    
    def get_best_class_accuracy(self):
        """
        Returns the class label with the highest accuracy in the best iteration.

        Returns
        -------
        best_class : str or int or None
            Class label with the highest accuracy in the best iteration.
            Returns None if no class scores are available.
        """
        if self.best_iteration_index_ == -1 or not self.iteration_performance_:
            self._verbose_out("No iterations performed or best iteration not found.")
            return None

        # self.iteration_performance_ stores [score, tree_size, class_scores_dict]
        class_scores = self.iteration_performance_[self.best_iteration_index_][2]
        if not class_scores:
            self._verbose_out("No class scores available for the best iteration.")
            return None

        best_score = -np.inf
        best_class = None
        for cls, score in class_scores.items():
            if score > best_score:
                best_score = score
                best_class = cls
        self._verbose_out(f"Best class accuracy is {best_score:.5f}; corresponds to Target={best_class}")
        return best_class

    def _verbose_out(self, *msg: str) -> None:
        """
        Print messages if the model is in verbose mode.
        """
        # Access the verbose attribute from the actual model, not ZuffyFitIterator itself
        if hasattr(self.model, "verbose") and self.model.verbose:
            for m in msg:
                print(m)