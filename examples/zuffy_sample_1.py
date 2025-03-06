import pandas as pd
from sklearn.datasets import load_iris
from zuffy.zuffy import ZuffyClassifier, functions, visuals
from zuffy.zuffy.wrapper import ZuffyFitIterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
targetClassNames = iris.target_names # we are given 0 = setosa, 1 = versicolor, 2 = virginica
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X)

zuffy = ZuffyClassifier(generations=15, verbose=1)
res = ZuffyFitIterator(zuffy, fuzzy_X, y, n_iter=10, split_at=0.25, random_state=231)

visuals.plot_evolution(
    res.getBestEstimator(),
    targetClassNames,
    res.getPerformance(),
    outputFilename='sample1_analysis')

visuals.graphviz_tree(
    res.getBestEstimator(),
    #targetNames,
    targetFeatureName = 'Species',
    targetClassNames=targetClassNames,
    featureNames=fuzzy_features_names,
    treeName=f"Iris Dataset (best accuracy: {res.getBestScore():.3f})",
    outputFilename='sample1_fpt_{res.getBestScore()*1000:.0f}')