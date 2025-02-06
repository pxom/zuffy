"""This file will just show how to write tests for the template classes."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from zuffy import ZuffyClassifier, functions, visuals
#from zuffy.visuals import ObjectColor, FeatureColor, OperatorColor, export_graphviz, graphviz_tree
#from zuffy import ZuffyClassifier #, TemplateEstimator, TemplateTransformer
from zuffy.functions import trimf

#import .functions
#from gplearn.genetic import SymbolicClassifier

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_classifier(data):
    """Check the internals and behaviour of `ZuffyClassifier`."""
    X, y = data
    #clf = ZuffyClassifier(SymbolicClassifier())
    clf = ZuffyClassifier()
    assert clf.const_range == ZuffyClassifier().const_range # "demo"
    #assert clf.demo_param == "function_set"

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
