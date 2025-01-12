"""
@author: Peter O'Mahony
This module contains the FPTGP Classifier and supporting methods and functions.
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause
#if __name__ == "__main__" and __package__ is None:
#    __package__ = "fptgp.fptgp"
#import sys    
#print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import StrOptions, Interval
import numbers # for scikit learn Interval

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from gplearn.genetic import SymbolicClassifier
from gplearn.utils import check_random_state
from gplearn.functions import _Function

from ._fpt_operators import *

# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class FPTGPClassifier(ClassifierMixin, BaseEstimator):
    """A Fuzzy Pattern Tree with Genetic Programming Classifier which uses gplearn to infer an FPT.

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
    >>> from skltemplate import TemplateClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = TemplateClassifier().fit(X, y)
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

    _parameter_constraints = {
        #"function_set" : ['tuple'],
        #"function_set" : [object],
        "function_set" :    ["array-like", object], # _Function],
        "n_components" :    [int, None],
        "population_size" : [Interval(numbers.Integral, 1, None, closed="left")],
        "generations" :     [int],
        "init_method" :     [StrOptions({'grow','full','half and half'})],
        "operator_only" :   [StrOptions({'all','operator_only'})],
    }

    default_function_set = [
                    COMPLEMENT,MAXIMUM,MINIMUM,
                    CONCENTRATOR,CONCENTRATOR2, #CONCENTRATOR3,CONCENTRATOR4,CONCENTRATOR8,
                    DILUTER,DILUTER2,
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
    #default_function_set = {4}

    def __init__(
                self,
                #*,
                n_components        = None,
                elite_size          = None,
                hall_of_fame        = None,
                #n_runs              = 1,
                multiclassifier     = 'OneVsRestClassifier',
                parsimony_object    = 'all',

                class_weight        = None,
                const_range         = None, # (-1., 1.),
                feature_names       = None,
                function_set        = default_function_set,
                generations         = 20,
                init_depth          = (2, 6),
                init_method         = 'half and half',
                low_memory          = False,
                max_samples         = 1.0,
                metric              = 'log loss',
                n_jobs              = 1,
                p_crossover         =0.9,
                p_hoist_mutation    =0.011,
                p_point_mutation    =0.01,
                p_point_replace     =0.05,
                p_subtree_mutation  =0.01,
                parsimony_coefficient=0.001,
                population_size     = 1000,
                random_state        =None,
                stopping_criteria   =0.0,
                tournament_size     =20,
                transformer         ='sigmoid',
                verbose             =0,
                warm_start          =False,
                ):
        
        self.n_components           = n_components
        self.elite_size             = elite_size
        self.hall_of_fame           = hall_of_fame
        #self.n_runs                 = n_runs
        self.multiclassifier        = multiclassifier
        self.parsimony_object       = parsimony_object
        
        self.class_weight           = class_weight
        self.const_range            = const_range
        self.feature_names          = feature_names
        self.function_set           = function_set
        self.generations            = generations
        self.init_depth             = init_depth
        self.init_method            = init_method
        self.low_memory             = low_memory
        self.max_samples            = max_samples
        self.metric                 = metric
        self.n_jobs                 = n_jobs
        self.p_crossover            = p_crossover
        self.p_hoist_mutation       = p_hoist_mutation
        self.p_point_mutation       = p_point_mutation
        self.p_point_replace        = p_point_replace
        self.p_subtree_mutation     = p_subtree_mutation
        self.parsimony_coefficient  = parsimony_coefficient
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
        X, y = self._validate_data(X, y)
        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        base_params = {
            'n_components':self.n_components, # these 4 are 'unexpected' - am I not using my modified gplearn?
            'elite_size':self.elite_size,
            'hall_of_fame':self.hall_of_fame,
            'parsimony_object':self.parsimony_object, # pom
            #'n_runs':self.n_runs,  # ???

            'class_weight':				self.class_weight,
            'const_range':				self.const_range,
            'feature_names':            self.feature_names,
            'function_set':				self.function_set,
            'generations':				self.generations,
            'init_depth':				self.init_depth,
            'init_method':				self.init_method,
            'low_memory':			    self.low_memory,
            'max_samples':			    self.max_samples,
            'metric':			        self.metric,
            'n_jobs':			        self.n_jobs,
            'p_crossover':			    self.p_crossover,
            'p_hoist_mutation':			self.p_hoist_mutation,
            'p_point_mutation':			self.p_point_mutation,
            'p_point_replace':			self.p_point_replace,
            'p_subtree_mutation':       self.p_subtree_mutation,
            'parsimony_coefficient':    self.parsimony_coefficient,
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
                    #verbose=self.verbose # not accepted by OneVsOneClassifier
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


class TemplateEstimator(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo_param"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

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
        X, y = self._validate_data(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return np.ones(X.shape[0], dtype=np.int64)


# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class TemplateTransformer(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, accept_sparse=True)

        # Return the transformer
        return self

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return np.sqrt(X)

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}

