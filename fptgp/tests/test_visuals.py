"""
Testing the FPTGP visual functions.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

from fptgp import FPTGPClassifier #, TemplateEstimator, TemplateTransformer
from fptgp.functions import trimf, fuzzify_col, fuzzify_data, flatten, fuzzy_feature_names, convert_to_numeric
from fptgp.visuals import ObjectColour, FeatureColour, OperatorColour, export_graphviz

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

def test_FeatureColour():
    featureColour = FeatureColour(['col1','col2','col3','col4','col5'])
    print(f'{featureColour.object_colours=}')

    tag = 'c0'
    nextCol = featureColour.getColour(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c1'
    nextCol = featureColour.getColour(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c0'
    nextCol = featureColour.getColour(tag)
    print(f'{tag=}  {nextCol=}')


def test_ObjectColour():
    objColour = ObjectColour()
    print(f'{objColour.object_colours=}')

    objColour = ObjectColour(['col1'])
    print(f'{objColour.object_colours=}')

    featureColour = FeatureColour(['col1','col2','col3','col4','col5'])
    print(f'{featureColour.object_colours=}')

    operatorColour = OperatorColour()
    print(f'{operatorColour.object_colours=}')

    nextCol = featureColour.getColour('bing')
    print(f'{nextCol=}')
    assert nextCol == featureColour.object_colours[0]

    nextCol = featureColour.getColour('bop')
    print(f'{nextCol=}')
    assert nextCol == featureColour.object_colours[1]

    nextCol = featureColour.getColour('bing')
    print(f'{nextCol=}')
    assert nextCol == featureColour.object_colours[0]

    print('---------------- LIST')
    for i in range(20):
        nextCol = featureColour.getColour(i)
        print(f'{nextCol=}')

def test_export_graphviz():
    X = [[1,2,3],[1,2,3],[1,2,3]]
    y = [0,1,2]
    fptgp = FPTGPClassifier(verbose=1)
    res   = fptgp.fit(X, y)

    program = res.estimators_
    print(program)
    res = export_graphviz(program, feature_names=None, fade_nodes=None, start=0, fillcolor='green')