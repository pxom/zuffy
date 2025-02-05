"""
Testing the FPTGP visual functions.
"""
 # Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from fptgp import FPTGPClassifier
from fptgp.visuals import ObjectColor, FeatureColor, OperatorColor, export_graphviz, graphviz_tree
from fptgp.wrapper import FPTGP_fit_iterator

def test_FeatureColor():
    #x = trimf(1, [1, 5, 9])
    featureColor = FeatureColor(['col1','col2','col3','col4','col5'])
    print(f'{featureColor.object_colors=}')

    tag = 'c0'
    nextCol = featureColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c1'
    nextCol = featureColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c0'
    nextCol = featureColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

def test_OperatorColor():
    #x = trimf(1, [1, 5, 9])
    operatorColor = OperatorColor(['col1','col2','col3','col4','col5'])
    print(f'{operatorColor.object_colors=}')

    tag = 'c0'
    nextCol = operatorColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c1'
    nextCol = operatorColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

    tag = 'c0'
    nextCol = operatorColor.getColor(tag)
    print(f'{tag=}  {nextCol=}')

def test_ObjectColor():
    objColor = ObjectColor()
    print(f'{objColor.object_colors=}')

    objColor = ObjectColor(['col1'])
    print(f'{objColor.object_colors=}')

    featureColor = FeatureColor(['col1','col2','col3','col4','col5'])
    print(f'{featureColor.object_colors=}')

    operatorColor = OperatorColor()
    print(f'{operatorColor.object_colors=}')

    nextCol = featureColor.getColor('bing')
    print(f'{nextCol=}')
    assert nextCol == featureColor.object_colors[0]

    nextCol = featureColor.getColor('bop')
    print(f'{nextCol=}')
    assert nextCol == featureColor.object_colors[1]

    nextCol = featureColor.getColor('bing')
    print(f'{nextCol=}')
    assert nextCol == featureColor.object_colors[0]

    print('---------------- LIST')
    for i in range(20):
        nextCol = featureColor.getColor(i)
        print(f'{nextCol=}')

def test_export_graphviz():
    X = [[1,2,2],[1,1,1],[2,2,2]]
    y = [0,1,2]
    fptgp = FPTGPClassifier(verbose=1, population_size=100, generations=5)
    est   = fptgp.fit(X, y)
    #res = FPTGP_fit_iterator(fptgp, X, y, n_iter=3, split_at=0.25, random_state=77)

    #program = res.estimators_
    #print(program)

    # why only send the first estimator?
    #tree = export_graphviz(res.estimators_[0], featureNames=None, fade_nodes=None, start=0, fillColor='green')
    source, graph = graphviz_tree(est)
    #tree = export_graphviz(est.estimators_[0]._program, featureNames=None, fade_nodes=None, start=0, fillColor='green')
    # verify res contains the expected string
    assert source[:9] == "digraph G"
