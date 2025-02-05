import pandas as pd
from sklearn.datasets import load_iris
from ..zuffy import ZuffyClassifier, functions, visuals
from ..zuffy.wrapper import ZuffyFitIterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
targetNames = iris.target_names
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X)

zuffy = ZuffyClassifier(generations=15, verbose=1)
res = ZuffyFitIterator(zuffy, fuzzy_X, y, n_iter=3, split_at=0.25, random_state=77)

visuals.plot_evolution(
    res.getBestEstimator(),
    targetNames,
    res.getPerformance(),
    outputFilename='sample1_analysis')

visuals.graphviz_tree(
    res.getBestEstimator(),
    targetNames,
    featureNames=fuzzy_features_names,
    treeName="Iris Dataset (best accuracy: " + str(round(res.getBestScore(),3)) + ")",
    outputFilename='sample1')