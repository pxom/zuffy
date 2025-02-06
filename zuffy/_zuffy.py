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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
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

# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class ZuffyClassifier(ClassifierMixin, BaseEstimator):
    """A Fuzzy Pattern Tree with Genetic Programming Classifier which uses gplearn to infer a FPT.

    Parameters
    ----------
    demo_param : str, default='demo'
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
    >>> from zuffy import ZuffyClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ZuffyClassifier().fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
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