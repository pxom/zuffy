import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from zuffy import ZuffyClassifier

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