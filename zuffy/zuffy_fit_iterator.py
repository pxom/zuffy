"""
This module contains the ZuffyFitIterator, a scikit-learn compatible
meta-estimator that repeatedly fits a classifier and builds Fuzzy 
Pattern Trees to find an optimal model.

"""

import copy
import numbers # for scikit learn Interval
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
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

    show_fuzzy_range : bool, default=True
        If `True`, the names of the output fuzzy features will include the
        numerical range (e.g., 'low feature_name (0.00 to 5.00)').
        If `False`, only the tag and feature name will be used (e.g., 'low feature_name').
        This parameter is passed directly to the `FuzzyTransformer`.

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
    best_estimator\\_ : object
        The best trained model (estimator) found across all iterations.
        This will be a Zuffy classifier or the `best_estimator_` from
        `GridSearchCV` if used.

    best_score\\_ : float
        The overall score achieved by the `best_estimator_` on its respective
        test set (accuracy).

    iteration_performance\\_ : list of tuples
        A list containing the performance metrics for each iteration.
        Each tuple contains:

        - `score` (float): The overall accuracy of the model on the test set.
        - `tree_size` (int): The total number of nodes in the Zuffy tree(s).
        - `class_scores_dict` (dict): A dictionary mapping class labels to
          their individual accuracy scores for that iteration.

    best_iteration_index\\_ : int
        The index of the iteration (0-based) that yielded the `best_estimator_`.

    smallest_tree_size\\_ : int
        The size (total number of nodes) of the tree in the `best_estimator_`.
        If `best_estimator_` is a `GridSearchCV`, this refers to the size of
        the best model found within `GridSearchCV`.

    fuzzy_feature_names\\_ : list of str
        The names of the features after fuzzification, derived from the
        `FuzzyTransformer` associated with the `best_estimator_`. This
        attribute is set during the `fit` method.

    feature_names_in\\_ : ndarray of str
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings (e.g., a Pandas DataFrame).

    n_features_in\\_ : int
        The number of features seen during `fit`.
    """

    # Using _parameter_constraints as per scikit-learn convention for validation
    _parameter_constraints: dict = {
        "tags": [list],
        "show_fuzzy_range": [bool],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "test_size": [Interval(numbers.Real, 0, 1, closed="both")],
        "random_state": ["random_state"],
    }

    @validate_params(
        _parameter_constraints,
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        model: Any,
        tags: List[str] = ['lo', 'med', 'hi'],
        show_fuzzy_range: bool = True,
        n_iter: int = 5,
        test_size: float = 0.2,
        random_state: Union[int, None] = None
        ):

        self.model = model
        if hasattr(self.model, "_validate_params"):
            self.model._validate_params()

        self.tags = tags
        self.show_fuzzy_range = show_fuzzy_range
        self.n_iter = n_iter
        self.test_size = test_size
        self.random_state = random_state

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"]
        },
        prefer_skip_nested_validation=True
    )
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: np.ndarray,
            feature_names: Union[List[str], None] = None,
            non_fuzzy: Union[List[str], None] = None
            ) -> "ZuffyFitIterator":
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
        X : array-like of shape (n_samples, n_features) or pd.DataFrame
            The input features (unfuzzified) to be used for training and testing.

        y : array-like of shape (n_samples,)
            The target labels corresponding to `X`.

        feature_names : list of str, optional
            A list of strings representing the names of the features in `X`.
            If `X` is a Pandas DataFrame and `feature_names` is None,
            `X.columns` will be used. If `X` is a NumPy array and `feature_names`
            is None, generic names (e.g., 'X0', 'X1') will be generated.

        non_fuzzy : list of str, optional
            A list of feature names that should NOT be fuzzified. These columns
            will be passed through directly to the model after one-hot encoding
            by the `FuzzyTransformer`.

        Returns
        -------
        self : ZuffyFitIterator
            The fitted instance of the ZuffyFitIterator.
        """
        # Set self.non_fuzzy early for use in _validate_data and FuzzyTransformer.
        self.non_fuzzy = non_fuzzy if non_fuzzy is not None else []

        # Preserve feature names if we get a DataFrame rather than a Numpy array
        if isinstance(X, pd.DataFrame) and feature_names is None:
            feature_names = X.columns

        if feature_names is None:
            # generate feature_names
            feature_names = [f'X{i}' for i in range(X.shape[1])]

        num_X = X.drop(self.non_fuzzy,axis=1)
        if num_X.shape[1] > 0:
            _, y = self._validate_data(num_X, y, accept_sparse=False, force_all_finite='allow-nan')
        else:
            print("Warning: There are no numeric columns in this dataset")
        self.feature_names_in_ = feature_names

        best_score_overall = -np.inf
        best_iter_idx = -1
        smallest_tree_size_overall = np.inf
        iteration_performance_list: List[Tuple[float, int, Dict[Any, float]]] = []
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
            #self._verbose_out(f"Class scores for iteration {i}: {class_scores}")

            # Determine the actual Zuffy estimator to calculate its tree size.
            # This handles both direct Zuffy models and GridSearchCV results.
            if isinstance(current_estimator, GridSearchCV):
                # Ensure the best_estimator_ exists if GridSearchCV didn't find a valid model
                if not hasattr(current_estimator, 'best_estimator_'):
                    self._verbose_out(f"Warning: GridSearchCV in iteration {i} "
                                      "did not find a best estimator.")
                    zuffy_estimator = None
                else:
                    zuffy_estimator = current_estimator.best_estimator_
            else:
                zuffy_estimator = current_estimator

            # Calculate the size of the model (assuming Zuffy models have a 'multi_' attribute)
            tree_size = 0
            if hasattr(zuffy_estimator, 'multi_') and \
               hasattr(zuffy_estimator.multi_, 'estimators_'):
                for e in zuffy_estimator.multi_.estimators_:
                    if hasattr(e, '_program'):
                        tree_size += len(e._program.program)
            self._verbose_out(f"Tree size at iteration {i}: {tree_size}")

            iteration_performance_list.append((score, tree_size, class_scores))

            # Update the best estimator based on score (primary) and then tree size (tie-breaker).
            if (score > best_score_overall) or \
               ((score == best_score_overall) and (tree_size < smallest_tree_size_overall)):
                best_iter_idx = i
                self.best_estimator_ = copy.copy(zuffy_estimator)
                self.fuzz_transformer_ = fuzz_transformer
                best_score_overall = score
                smallest_tree_size_overall = tree_size
                self._verbose_out(f"New best estimator found: Iteration {i} with score "
                                  f"{score:.5f} and tree size {tree_size}")

            iter_duration = round(time.time() - iter_start_time, 1)
            avg_score_so_far = sum_scores / (i + 1)
            self._verbose_out(f"Iteration #{i} took {iter_duration}s | "
                              f"Best accuracy so far: {best_score_overall:.5f} "
                              f"(size: {smallest_tree_size_overall}) | "
                              f"Average score: {avg_score_so_far:.5f}")

        self._verbose_out(f"Finished iterating. Best iteration: {best_iter_idx}")
        self.best_score_ = best_score_overall
        self.iteration_performance_ = iteration_performance_list
        self.best_iteration_index_ = best_iter_idx
        self.smallest_tree_size_ = smallest_tree_size_overall
        return self

    def _perform_single_fit_job(
            self,
            model: Any,
            X: Union[np.ndarray, pd.DataFrame], # Input can be DataFrame or NumPy array
            y: np.ndarray,
            test_size: float = 0.2,
            random_state: Union[int, None] = None
            ) -> Tuple[float, Any, Dict[Any, float], FuzzyTransformer]:
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

        fuzz_transformer : FuzzyTransformer
            The fitted `FuzzyTransformer` used in this specific iteration.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y, random_state=random_state
        )

        fuzz_transformer = FuzzyTransformer(
            feature_names=self.feature_names_in_,
            non_fuzzy=self.non_fuzzy,
            tags=self.tags,
            show_fuzzy_range=self.show_fuzzy_range
            )
        fuzz_transformer.fit(X_train)
        fuzzy_X_train = fuzz_transformer.transform(X_train)
        self.fuzzy_feature_names_ = fuzz_transformer.feature_names_out_

        # Fit the model (ZuffyClassifier or GridSearchCV) on the fuzzified training data
        fitted_model = model.fit(fuzzy_X_train, y_train)
        # Transform the test data using the fitted transformer
        fuzzy_X_test = fuzz_transformer.transform(X_test)
        # Evaluate the model on the fuzzified test data
        score = fitted_model.score(fuzzy_X_test, y_test)
        self._verbose_out(f"Overall test score: {score:.8f}")

        # Get predictions to calculate per-class accuracies
        predictions = fitted_model.predict(fuzzy_X_test)
        class_scores = {}
        for cls in np.unique(y_test):
            idx = (y_test == cls)
            if np.any(idx):  # Ensure there are samples for the current class in the test split
                class_accuracy = accuracy_score(y_test[idx], predictions[idx])
                class_scores[cls] = class_accuracy
                self._verbose_out(f"Accuracy for class {cls}: {class_accuracy:.5f}")
            else:
                self._verbose_out(f"Class {cls} not present in this test split.")

        #avg_score = round(np.mean(list(class_scores.values())), 5) if class_scores else 0.0
        #self._verbose_out(f"Average Class score: {avg_score}")

        return score, fitted_model, class_scores, fuzz_transformer

    def get_best_class_accuracy(self) -> Union[str, int, None]:
        """
        Returns the class label with the highest accuracy in the best iteration.

        Returns
        -------
        best_class : str or int or None
            Class label with the highest accuracy in the best iteration.
            Returns None if no class scores are available.
        """
        if self.best_iteration_index_ == -1 or not hasattr(self, 'iteration_performance_') or \
           not self.iteration_performance_:
            self._verbose_out("No iterations performed or best iteration not found.")
            return None

        # self.iteration_performance_ stores (score, tree_size, class_scores_dict)
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
        self._verbose_out(f"Best class accuracy is {best_score:.5f}; "
                          f"corresponds to Target={best_class}")
        return best_class

    def _verbose_out(self, *msg: str) -> None:
        """
        Prints messages if the model's 'verbose' attribute is True.

        This method checks the 'verbose' attribute on the *contained* model
        (e.g., ZuffyClassifier or GridSearchCV) to control output.
        """
        # Access the verbose attribute from the actual model, not ZuffyFitIterator itself
        if hasattr(self.model, "verbose") and self.model.verbose:
            for m in msg:
                print(m)
