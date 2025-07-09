"""
Support function for generating a Decision Tree for comparison.

"""

import numbers
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils._param_validation import Interval, validate_params

@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
        "features": [None, list],
        "output_filename": [str],
        "max_leaf_nodes": [Interval(numbers.Integral, 1, None, closed='left'), None],
        "random_state": [numbers.Integral, None]
    },
    prefer_skip_nested_validation=True
)
def do_model_dt(X: np.ndarray, y: np.ndarray, features: Optional[List[str]],
                output_filename: str, source_filename: str = None,
                max_leaf_nodes: Optional[int] = 10, random_state: Optional[int] = None
                ) -> DecisionTreeClassifier:
    """
    Trains a Decision Tree Classifier model and exports its visualization.

    This function builds a `DecisionTreeClassifier` using the provided data and
    hyperparameters. It then generates and saves a visualization of the
    trained tree using `sklearn.tree.plot_tree` and prints a text-based
    representation. This can be used for comparison with Fuzzy Pattern Trees.

    Parameters
    ----------
    X : numpy.ndarray
        The input features data, used for training the decision tree.
    y : numpy.ndarray
        The target variable data, used for training the decision tree.
    features : list of str, optional
        A list of string names for the features. If `None`, default feature
        names will be used by `plot_tree` (e.g., 'X[0]', 'X[1]').
        Defaults to `None`.
    output_filename : str
        The full path and filename (e.g., 'path/to/dt_plot.png') to save the
        decision tree plot.
    max_leaf_nodes : int, optional
        The maximum number of leaf nodes for the decision tree. This parameter
        helps control the complexity and prevent overfitting. Defaults to 10.
    random_state : int, optional
        Controls the randomness of the estimator's splitting process.
        - Pass an `int` for reproducible output across multiple function calls.
        - Pass `None` for a different random state each time.
        Defaults to 221.

    Returns
    -------
    est_dt : sklearn.tree.DecisionTreeClassifier
        The trained `DecisionTreeClassifier` model instance.
    """
    est_dt = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state
    )
    est_dt.fit(X, y)

    _, ax = plt.subplots(figsize=(14, 14)) # Using larger figsize to prevent squashed trees
    tree.plot_tree(est_dt, feature_names=features, filled=True, rounded=True, fontsize=9, ax=ax)

    plt.title('Decision Tree using Fuzzified Dataset')

    # export_text prints to console by default; capturing it for potential use or logging
    tree_text_representation = tree.export_text(est_dt, feature_names=features)
    print('\nDecision Tree Text Representation:\n', tree_text_representation)

    if source_filename is not None:
        export_graphviz(est_dt, out_file=source_filename, feature_names=features, filled=True)

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()
    plt.close() # Close the figure to free memory

    return est_dt
