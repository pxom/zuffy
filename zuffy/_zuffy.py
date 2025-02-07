"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
This module contains the zuffy Classifier and supporting methods and functions.
"""

#if __name__ == "__main__" and __package__ is None:
#    __package__ = "zuffy.zuffy"
#import sys    
#print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

import numbers # for scikit learn Interval
import numpy as np
import sklearn # so that we can check the version number
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils._param_validation import StrOptions, Interval, Options
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data # when running on ubuntu with scikit-learn >= 1.6.0
except ImportError:
    #print('could not import sklearn.utils.validation.validate_data because it is not available. scikit-learn is probably <1.6.0')
    pass


from gplearn.functions import _Function
from gplearn.genetic import SymbolicClassifier
from gplearn.utils import check_random_state

from ._fpt_operators import *

class ZuffyClassifier(ClassifierMixin, BaseEstimator):
    """A Fuzzy Pattern Tree with Genetic Programming Classifier which uses gplearn to infer a FPT.

    These parameters are passed through to the gplearn SymbolicClassifier and further documentation is 
    available on that at https://gplearn.readthedocs.io/en/stable/reference.html#gplearn.genetic.SymbolicClassifier.

    This classifier uses OneVsRestClassifier classifier to handle multi-class classifications.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'MAXIMUM' : tbd, arity=2.
        - 'MINIMUM' : tbd, arity=2.
        - 'COMPLEMENT' : Not, arity=2.

    transformer : str, optional (default='sigmoid')
        The name of the function through which the raw decision function is
        passed. This function will transform the raw decision function into
        probabilities of each class.

        This can also be replaced by your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

    metric : str, optional (default='log loss')
        The name of the raw fitness metric. Available options include:

        - 'log loss' aka binary cross-entropy loss.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    class_weight : dict, 'balanced' or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    gplearn

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.



    Parameters
    ----------
    class_weight : 'balanced', list or None (default)
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from zuffy.zuffy import ZuffyClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ZuffyClassifier().fit(X, y)
    >>> clf.predict(X)
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.

    _parameter_constraints = { # POM These are only checked when we use the fit method - not checked when initialising - why?
        "class_weight":         [StrOptions({'balanced'}), list, None],
        "const_range":          [None, list],
        #"elite_size":           [Interval(numbers.Integral, 1, None, closed="left"), None],
        "feature_names":        [list, None],
        "function_set":         [list],
        "generations":          [Interval(numbers.Integral, 1, None, closed="left")],
        #"hall_of_fame":         [Interval(numbers.Integral, 1, None, closed="left"), None],
        "init_depth":           [tuple],
        "init_method":          [StrOptions({'half and half','grow','full'})],
        "low_memory":           [bool],
        "max_samples":          [Interval(numbers.Real, 0, 1, closed="both")],
        "metric":               [StrOptions({'log loss'})], # needs to allow for a custom metric
        "multiclassifier":      [StrOptions({'OneVsRestClassifier','OneVsOneClassifier'})],
        #"n_components":         [Interval(numbers.Integral, 1, None, closed="left"), None],
        "n_jobs":               [Interval(numbers.Integral, 1, None, closed="left")],
        "p_crossover":          [Interval(numbers.Real, 0, 1, closed="both")],
        "p_hoist_mutation":     [Interval(numbers.Real, 0, 1, closed="both")],
        "p_point_mutation":     [Interval(numbers.Real, 0, 1, closed="both")],
        "p_point_replace":      [Interval(numbers.Real, 0, 1, closed="both")],
        "p_subtree_mutation":   [Interval(numbers.Real, 0, 1, closed="both")],
        "parsimony_coefficient":[Interval(numbers.Real, 0, 1, closed="both")],
        #"parsimony_object":     [StrOptions({'all','operator_only','ratio'})],
        "population_size":      [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state":         ["random_state"],
        "stopping_criteria":    [Interval(numbers.Real, 0, 1, closed="both")],
        "tournament_size":      [Interval(numbers.Integral, 1, None, closed="left")],
        "transformer":          [StrOptions({'sigmoid'})],
        "verbose":              ["verbose"],
        "warm_start":           [bool],
    }
    '''
            "estimator": [HasMethods("fit")],
            "threshold": [Interval(Real, None, None, closed="both"), str, None],
            "prefit": ["boolean"],
            "norm_order": [
                Interval(Integral, None, -1, closed="right"),
                Interval(Integral, 1, None, closed="left"),
                Options(Real, {np.inf, -np.inf}),
            ],
            "max_features": [Interval(Integral, 0, None, closed="left"), callable, None],
            "importance_getter": [str, callable],

        "n_jobs": [None, Integral],

        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "precompute": ["boolean", "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "selection": [StrOptions({"cyclic", "random"})], 

        "eps": [Interval(Real, 0, None, closed="neither")],
        "n_alphas": [Interval(Integral, 1, None, closed="left")],
        "alphas": ["array-like", None],
        "fit_intercept": ["boolean"],
        "precompute": [StrOptions({"auto"}), "array-like", "boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "cv": ["cv_object"],
        "verbose": ["verbose"],
    '''
    default_function_set = [
                    COMPLEMENT,MAXIMUM,MINIMUM,
                    # DILUTER, CONCENTRATOR, #CONCENTRATOR2, CONCENTRATOR3,CONCENTRATOR4,CONCENTRATOR8, DILUTER2,
                    # DIFFUSER,INTENSIFIER,
                    #WA_P1,
                    #WA_P2,
                    #WA_P3,
                    #WA_P4,
                    #WA_P5,
                    #WA_P6,
                    #WA_P7,
                    #WA_P8,
                    #WA_P9,
                    #OWA_P1,
                    #OWA_P2,
                    #OWA_P3,
                    #OWA_P4,
                    #OWA_P5,
                    #OWA_P6,
                    #OWA_P7,
                    #OWA_P8,
                    #OWA_P9
                ]

    def __init__(
                self,
                class_weight          = None,
                const_range           = None,
                #elite_size            = None,
                feature_names         = None,
                function_set          = default_function_set,
                generations           = 20,
                #hall_of_fame          = None,
                init_depth            = (2, 6),
                init_method           = 'half and half',
                low_memory            = False,
                max_samples           = 1.0,
                metric                = 'log loss',
                multiclassifier       = 'OneVsRestClassifier',
                #n_components          = None,
                n_jobs                = 1,
                p_crossover           = 0.9,
                p_hoist_mutation      = 0.011,
                p_point_mutation      = 0.01,
                p_point_replace       = 0.05,
                p_subtree_mutation    = 0.01,
                parsimony_coefficient = 0.001,
                #parsimony_object      =  'all',
                population_size       =  1000,
                random_state          = None,
                stopping_criteria     = 0.0,
                tournament_size       = 20,
                transformer           = 'sigmoid',
                verbose               = 0,
                warm_start            = False,
                ):
        
        self.class_weight           = class_weight
        self.const_range            = const_range
        #self.elite_size             = elite_size
        self.feature_names          = feature_names
        self.function_set           = function_set
        self.generations            = generations
        #self.hall_of_fame           = hall_of_fame
        self.init_depth             = init_depth
        self.init_method            = init_method
        self.low_memory             = low_memory
        self.max_samples            = max_samples
        self.metric                 = metric
        self.multiclassifier        = multiclassifier
        #self.n_components           = n_components
        self.n_jobs                 = n_jobs
        self.p_crossover            = p_crossover
        self.p_hoist_mutation       = p_hoist_mutation
        self.p_point_mutation       = p_point_mutation
        self.p_point_replace        = p_point_replace
        self.p_subtree_mutation     = p_subtree_mutation
        self.parsimony_coefficient  = parsimony_coefficient
        #self.parsimony_object       = parsimony_object
        self.population_size        = population_size
        self.random_state           = random_state
        self.stopping_criteria      = stopping_criteria
        self.tournament_size        = tournament_size
        self.transformer            = transformer
        self.verbose                = verbose
        self.warm_start             = warm_start


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.

        if sklearn.__version__ < '1.6.0': # too many issues with OvR etc so require sklearn < 1.6.0
            X, y = self._validate_data(X, y)
            # We need to make sure that we have a classification task
            #check_classification_targets(y)
        else:
            X, y = validate_data(X, y)
            #pass
        

        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        base_params = {
            'class_weight':				self.class_weight,
            'const_range':				self.const_range,
            #'elite_size':               self.elite_size,
            'feature_names':            self.feature_names,
            'function_set':				self.function_set,
            'generations':				self.generations,
            #'hall_of_fame':             self.hall_of_fame,
            'init_depth':				self.init_depth,
            'init_method':				self.init_method,
            'low_memory':			    self.low_memory,
            'max_samples':			    self.max_samples,
            'metric':			        self.metric,
            #'n_components':             self.n_components,
            'n_jobs':			        self.n_jobs,
            'p_crossover':			    self.p_crossover,
            'p_hoist_mutation':			self.p_hoist_mutation,
            'p_point_mutation':			self.p_point_mutation,
            'p_point_replace':			self.p_point_replace,
            'p_subtree_mutation':       self.p_subtree_mutation,
            'parsimony_coefficient':    self.parsimony_coefficient,
            #'parsimony_object':         self.parsimony_object,
            'population_size':			self.population_size,
            'random_state':			    self.random_state,
            'stopping_criteria':		self.stopping_criteria,
            'tournament_size':			self.tournament_size,
            'transformer':			    self.transformer,
            'verbose':			        self.verbose,
            'warm_start':			    self.warm_start
            }

        if self.multiclassifier=='OneVsOneClassifier':
            ovr = OneVsOneClassifier( # OneVsRestClassifier( # 
                    SymbolicClassifier(**base_params),
                    )
        elif self.multiclassifier=='OneVsRestClassifier':
            ovr = OneVsRestClassifier( 
                    SymbolicClassifier(**base_params),
                    verbose=self.verbose
                    )
        else:
            raise ValueError('multiclassifier must be one of: '
                             f'OneVsOneClassifier, OneVsRestClassifier. Found {self.multiclassifier}')

        #sym = SymbolicClassifier(**base_params)
        return ovr.fit(X,y)
        # Return the classifier - this is required by scikit-learn standard!!!!!!!!!!!! tests pass if we return self
        #return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, reset=False)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]