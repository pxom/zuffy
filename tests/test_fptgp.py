"""
Testing the FPTGP functions.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

from fptgp import FPTGPClassifier #, TemplateEstimator, TemplateTransformer
from fptgp.functions import trimf, fuzzify_col, fuzzify_data, flatten, fuzzy_feature_names, convert_to_numeric

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def pom(x):
    print('pom',x)
    return x+2
    
def test_fptgp_pom(): # simple non-FPTGP test to test tests
    x = 3
    print('test_pom:1',x)
    assert pom(x) == 5

def test_fptgp_pom2():
    x = 2
    print('test_pom:2',x)
    assert pom(x) == 4  

def test_fptgp_trimf1():
    x = np.array([1.0])
    min = 1
    mid = 5
    max = 9
    lo = trimf(x, [min, min, mid])
    md = trimf(x, [min, mid, max])
    hi = trimf(x, [mid, max, max])
    z = [lo, md, hi]

    print('result of trimf is',z)
    expected = np.array([[1.]  , [0.0], [0.0]])
    assert np.array_equal(z, expected)

def test_fptgp_trimf2():
    x = np.array([5.0])
    min = 1
    mid = 5
    max = 9
    lo = trimf(x, [min, min, mid])
    md = trimf(x, [min, mid, max])
    hi = trimf(x, [mid, max, max])
    z = [lo, md, hi]

    print('result of trimf is',z)
    expected = np.array([[0.]  , [1.0], [0.0]])
    assert np.array_equal(z, expected)

@pytest.mark.parametrize("x,min,mid,max,expected",[
    [7,1,5, 9,[[0.],[0.5], [0.5]]],
    [6,1,5, 9,[[0.],[0.75],[0.25]]],
    [6,1,5,15,[[0.],[0.9], [0.1]]],
    ])
def test_fptgp_trimf3(x,min,mid,max,expected):
    x = np.array([x])
    lo = trimf(x, [min, min, mid])
    md = trimf(x, [min, mid, max])
    hi = trimf(x, [mid, max, max])
    z = [lo, md, hi]

    print('result of trimf is',z)
    assert np.array_equal(z, expected)

@pytest.mark.skip
def test_fptgp_trimf4():
    x = np.array([7.0])
    min = 1
    mid = 5
    max = 9
    lo = trimf(x, [-1, 0, min, min, mid])
    md = trimf(x, [min, mid, max])
    hi = trimf(x, [mid, max, max])
    z = [lo, md, hi]

    print('result of trimf is',z)
    expected = np.array([[0.]  , [0.5], [0.5]])
    assert np.array_equal(z, expected)

#@pytest.mark.skip
def test_fuzzify_col1():
    col = np.array([1.0,2.0,3.0])
    feature_name = 'bang'
    info = True
    tags = ['bip','bop','burp']

    #expected = ([np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])], None, [2.0, 2.0, 2.0])
    #expected = ([np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])], None)
    #expected = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    expected = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    #expected = ([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], None)

    res, dummy = fuzzify_col(col, feature_name, info=False, tags=None)
    print('res=',res)
    print('dummy=',dummy)
    print('expected=',expected)

    assert np.array_equal(res, expected)

@pytest.mark.parametrize("col,expected",[
    [np.array([1.0,2.0,3.0]), 
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]],
    [np.array([10.0,20.0,30.0]), 
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]],
    ])
def test_fuzzify_col(col, expected):
    #col = np.array([1.0,2.0,3.0])
    feature_name = 'bang'
    info = True
    tags = ['bip','bop','burp']

    #expected = ([np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])], None, [2.0, 2.0, 2.0])
    #expected = ([np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])], None)
    #expected = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    expected = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    
    res, dummy = fuzzify_col(col, feature_name, info=False, tags=None)
    print('res=',res)
    print('dummy=',dummy)
    print('expected=',expected)

    assert np.array_equal(res, expected)

@pytest.mark.parametrize("data,exp_data,exp_cols",[
    [[1,2,3],    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], ['bip zip| (1 to 2)', 'bop zip| (1 to 2 to 3)', 'burp zip| (2 to 3)']], 
    [[10,20,30], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], ['bip zip| (10 to 20)', 'bop zip| (10 to 20 to 30)', 'burp zip| (20 to 30)']], 
    ])
def test_fuzzify_data(data, exp_data, exp_cols):
    data = pd.DataFrame(data)
    data.columns = ['zip']
    non_fuzzy = []
    info = False
    tags = ['bip','bop','burp']
    #exp_data = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    #exp_cols = ['bip zip| (1 to 2)', 'bop zip| (1 to 2 to 3)', 'burp zip| (2 to 3)']

    res, cols = fuzzify_data(data, non_fuzzy, info=info, tags=tags)
    print('res=',res)
    print('cols=',cols)

    print('exp_data=',exp_data)
    print('exp_cols=',exp_cols)
    assert np.array_equal(cols, exp_cols)
    assert np.array_equal(res, exp_data)

def test_flatten():
    x = [[1,2,3],[4],[66,66,66],[],[5]]
    z = flatten(x)
    assert z == [1,2,3,4,66,66,66,5]

@pytest.mark.parametrize("flist, tags, expected",[
    [['hello','world'],['1st','2nd'],['1st hello', '2nd hello', '1st world', '2nd world']],
    ])
def test_fuzzy_feature_names(flist, tags, expected):
    res  = fuzzy_feature_names(flist, tags)
    assert res == expected

@pytest.mark.parametrize("my_data, target, exp_target_classes, exp_data",[
    [['red','green','blue','red','red','red','red'],0,['blue', 'green', 'red'],[[2],[1],[0],[2],[2],[2],[2]] ],
    ])
def test_convert_to_numeric(my_data, target, exp_target_classes, exp_data):
    #my_data = ['red','green','blue','red','red','red','red']
    my_data = pd.DataFrame(my_data)
    #target = 0 #'bang'
    print(f"-=-=-=")
    print(f"{my_data=}")
    print(f"-=-=-=")
    target_classes, new_data = convert_to_numeric(my_data, target)
    print(f"{target_classes=}")
    #new_data = list(new_data)
    print(f"-=-=-=")
    print(f"{new_data.values=}")
    assert np.array_equal(target_classes, exp_target_classes)
    assert np.array_equal(new_data, exp_data)

def test_fptgp_classifier_checks():
    """Run the sklearn estimator validation checks on FPTGP"""

    check_estimator(FPTGPClassifier(population_size=1000,
                                      generations=5))
