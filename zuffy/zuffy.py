"""
This module contains the Zuffy Classifier and supporting methods and functions.

"""


import numbers # for scikit learn Interval
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils._param_validation import StrOptions, Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from gplearn.genetic import SymbolicClassifier

from ._fpt_operators import COMPLEMENT, MAXIMUM, MINIMUM

class ZuffyClassifier(ClassifierMixin, BaseEstimator):
    """A Fuzzy Pattern Tree Classifier which uses genetic programming to infer the model structure.

    This classifier wraps ``gplearn.genetic.SymbolicClassifier`` and handles multi-class 
    classification using ``sklearn.multiclass.OneVsRestClassifier`` or 
    ``sklearn.multiclass.OneVsOneClassifier``. It uses the OneVsRestClassifier classifier to 
    handle multi-class classifications by default.

    These parameters are passed through to the gplearn SymbolicClassifier and further documentation
    is available on that at
    https://gplearn.readthedocs.io/en/stable/reference.html#gplearn.genetic.SymbolicClassifier.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    const_range : tuple of two floats, or None, default=None
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    feature_names : list, optional, default=None
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    function_set : iterable, default=(COMPLEMENT, MAXIMUM, MINIMUM)
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate individual functions (from gplearn's
        built-in set) or custom functions created using `make_function` from
        `gplearn.functions`.

    generations : int, default=20
        The number of generations to evolve.

    init_depth : tuple of two ints, default=(2, 10)
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : {'half and half', 'grow', 'full'}, default='half and half'
        The method to use for initializing the population.

        - 'grow': Nodes are chosen at random from both functions and terminals,
          allowing for smaller trees than `init_depth` allows. Tends to grow
          asymmetrical trees.
        - 'full': Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half': Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    low_memory : bool, default=False
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    max_samples : float, default=1.0
        The fraction of samples to draw from X to evaluate each program on.

    metric : str or callable, default='log loss'
        The name of the raw fitness metric. Available options include:
        - 'log loss' aka binary cross-entropy loss.
        Can also be a custom callable metric.

    multiclassifier : {'OneVsRestClassifier', 'OneVsOneClassifier'}, default='OneVsRestClassifier'
        The strategy to use for handling multi-class classification problems.

    n_jobs : int, default=1
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    p_crossover : float, default=0.9
        The probability of performing crossover on a tournament winner.

    p_hoist_mutation : float, default=0.01
        The probability of performing hoist mutation on a tournament winner.

    p_point_mutation : float, default=0.01
        The probability of performing point mutation on a tournament winner.

    p_point_replace : float, default=0.05
        For point mutation only, the probability that any given node will be
        mutated.

    p_subtree_mutation : float, default=0.01
        The probability of performing subtree mutation on a tournament winner.

    parsimony_coefficient : float or "auto", default=0.001
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. If "auto" the
        parsimony coefficient is recalculated for each generation.

    population_size : int, default=1000
        The number of programs in each generation.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for reproducibility.

    stopping_criteria : float, default=0.0
        The required metric value required in order to stop evolution early.

    tournament_size : int, default=20
        The number of programs that will compete to become part of the next
        generation.

    transformer : str or callable, default='sigmoid'
        The name of the function through which the raw decision function is
        passed. This function will transform the raw decision function into
        probabilities of each class. Can also be a custom callable transformer.

    verbose : int, default=0
        Controls the verbosity of the evolution building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    Attributes
    ----------
    classes\\_ : ndarray of shape (n_classes,)
        The unique class labels observed in `y`.

    multi\\_ : OneVsRestClassifier or OneVsOneClassifier
        The underlying scikit-learn multi-class classifier used for training.

    n_features_in\\_ : int
        Number of features seen during `fit`.

    feature_names_in\\_ : ndarray of str, shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    gplearn.genetic.SymbolicClassifier : The core genetic programming classifier.
    sklearn.multiclass.OneVsRestClassifier : Multi-class strategy.
    sklearn.multiclass.OneVsOneClassifier : Multi-class strategy.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from zuffy.zuffy import ZuffyClassifier # Assuming zuffy is installed and on path
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ZuffyClassifier(random_state=42).fit(X, y)
    >>> predictions = clf.predict(X)
    >>> print(predictions.shape)
    (150,)

    """

    # pylint: disable=too-many-instance-attributes
    # We need this for gplearn compatibility

    _parameter_constraints = {
        # The domain for the parameters is mostly defined by the gplearn classifier
        "class_weight": [StrOptions({'balanced'}), dict, None],
        "const_range": [None, tuple],
        "feature_names": [list, None],
        "function_set": ["array-like"], # Expecting list of _Function objects or strings
        "generations": [Interval(numbers.Integral, 1, None, closed="left")],
        "init_depth": [tuple],
        "init_method": [StrOptions({'half and half','grow','full'})],
        "low_memory": [bool],
        "max_samples": [Interval(numbers.Real, 0, 1, closed="both")],
        "metric": [StrOptions({'log loss'}), callable], # Allow for a custom metric
        "multiclassifier": [StrOptions({'OneVsRestClassifier','OneVsOneClassifier'})],
        "n_jobs": [Interval(numbers.Integral, -1, None, closed="left")],
        "p_crossover": [Interval(numbers.Real, 0, 1, closed="both")],
        "p_hoist_mutation": [Interval(numbers.Real, 0, 1, closed="both")],
        "p_point_mutation": [Interval(numbers.Real, 0, 1, closed="both")],
        "p_point_replace": [Interval(numbers.Real, 0, 1, closed="both")],
        "p_subtree_mutation": [Interval(numbers.Real, 0, 1, closed="both")],
        "parsimony_coefficient": [Interval(numbers.Real, 0, 1, closed="both"),
                                   StrOptions({"auto"})],
        "population_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "stopping_criteria": [Interval(numbers.Real, 0, None, closed="both")],
        "tournament_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "transformer": [StrOptions({'sigmoid'}), callable],
        "verbose": [numbers.Integral, bool],
        "warm_start": [bool],
    }

    default_function_set = (
        COMPLEMENT,
        MAXIMUM,
        MINIMUM
    )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-locals
    # We need this for gplearn compatibility

    def __init__(
        self,
        class_weight=None,
        const_range=None,
        feature_names=None,
        function_set=default_function_set,
        generations=20,
        init_depth=(2, 10),
        init_method='half and half',
        low_memory=False,
        max_samples=1.0,
        metric='log loss',
        multiclassifier='OneVsRestClassifier',
        n_jobs=1,
        p_crossover=0.9,
        p_hoist_mutation=0.01,
        p_point_mutation=0.01,
        p_point_replace=0.05,
        p_subtree_mutation=0.01,
        parsimony_coefficient=0.001,
        population_size=1000,
        random_state=None,
        stopping_criteria=0.0,
        tournament_size=20,
        transformer='sigmoid',
        verbose=0,
        warm_start=False,
        ):
        """Initialize ZuffyClassifier.

        All parameters are passed directly to the underlying `gplearn.genetic.SymbolicClassifier`
        or `sklearn.multiclass` wrappers. Refer to their respective documentations for
        detailed explanations.
        """

        self.class_weight = class_weight
        self.const_range = const_range
        self.feature_names = feature_names
        self.function_set = function_set
        self.generations = generations
        self.init_depth = init_depth
        self.init_method = init_method
        self.low_memory = low_memory
        self.max_samples = max_samples
        self.metric = metric
        self.multiclassifier = multiclassifier
        self.n_jobs = n_jobs
        self.p_crossover = p_crossover
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.p_subtree_mutation = p_subtree_mutation
        self.parsimony_coefficient = parsimony_coefficient
        self.population_size = population_size
        self.random_state = random_state
        self.stopping_criteria = stopping_criteria
        self.tournament_size = tournament_size
        self.transformer = transformer
        self.verbose = verbose
        self.warm_start = warm_start

        self.multi_ = None
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the Fuzzy Pattern Tree Classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values. An array of int or str.

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It runs different checks on the input data and defines some attributes associated
        # with the input data: `n_features_in_` and `feature_names_in_`.

        X, y = self._validate_data(X, y)

        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the fuzzified training data for use later (perhaps by feature importance evaluation)
        self.X_ = X
        self.y_ = y

        base_params = {
            'class_weight': self.class_weight,
            'const_range': self.const_range,
            'feature_names': self.feature_names,
            'function_set': self.function_set,
            'generations': self.generations,
            'init_depth': self.init_depth,
            'init_method': self.init_method,
            'low_memory': self.low_memory,
            'max_samples': self.max_samples,
            'metric': self.metric,
            'n_jobs': self.n_jobs,
            'p_crossover': self.p_crossover,
            'p_hoist_mutation': self.p_hoist_mutation,
            'p_point_mutation': self.p_point_mutation,
            'p_point_replace': self.p_point_replace,
            'p_subtree_mutation': self.p_subtree_mutation,
            'parsimony_coefficient': self.parsimony_coefficient,
            'population_size': self.population_size,
            'random_state': self.random_state,
            'stopping_criteria': self.stopping_criteria,
            'tournament_size': self.tournament_size,
            'transformer': self.transformer,
            'verbose': 0 if self.multiclassifier=='OneVsOneClassifier' else self.verbose,
            'warm_start': self.warm_start
            }

        if self.multiclassifier=='OneVsOneClassifier':
            self.multi_ = OneVsOneClassifier(
                    SymbolicClassifier(**base_params),
                    n_jobs=self.n_jobs,
                    )
        elif self.multiclassifier=='OneVsRestClassifier':
            self.multi_ = OneVsRestClassifier(
                    SymbolicClassifier(**base_params),
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                    )
        else:
            raise ValueError('multiclassifier must be one of: '
                             'OneVsOneClassifier, OneVsRestClassifier. '
                             f'Found {self.multiclassifier}')

        self.multi_.fit(X,y)
        # Return the classifier - this is required by scikit-learn fit method
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class labels for each sample.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation for X. Use check_array for prediction inputs.
        X = self._validate_data(X, reset=False) # `reset=False` to preserve `n_features_in_` etc.

        return self.multi_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities for each sample.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        # Ensure the number of features matches what was seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this ZuffyClassifier was "
                f"fitted with {self.n_features_in_} features."
            )

        # Delegate probability prediction to the underlying multi-class classifier
        return self.multi_.predict_proba(X)
