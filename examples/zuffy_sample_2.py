import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from zuffy.zuffy import ZuffyClassifier, functions, visuals
from zuffy.zuffy.zwrapper import ZuffyFitIterator
import zuffy.zuffy._fpt_operators 
from zuffy.zuffy._fpt_operators import MAXIMUM, MINIMUM, COMPLEMENT, CONCENTRATOR, DILUTER
from gplearn.functions import _Function

def _relu(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return (np.maximum(0, x1)) 

relu = _Function(function=_relu, name='relu', arity=1)

def _tanh(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return np.tanh(x1) 

tanh = _Function(function=_tanh, name='tanh', arity=1)

my_data = pd.read_csv('../datasets/spaceshipTitanic_train.csv', sep=',', header=0, skiprows=0)

target_name = 'Transported'
X = my_data.drop(target_name, axis=1)
X.drop(['PassengerId','Name'], inplace=True, axis=1)
y = my_data[target_name]
target_classes = y.unique().tolist()

# if target_classes are False, True then the values should be 0, 1

fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X, non_fuzzy=['HomePlanet','CryoSleep','Destination','VIP'])

zuffy = ZuffyClassifier(
            generations=50,
            population_size=350,
            tournament_size=60,
            init_depth=(10,16),
            transformer=tanh,
            function_set=[MAXIMUM, MINIMUM, COMPLEMENT], #, CONCENTRATOR, DILUTER],
            parsimony_coefficient=0.0003,
            p_crossover=0.6,
            p_subtree_mutation=0.2,
            p_hoist_mutation=0.09,
            p_point_mutation=0.01,
            #stopping_criteria=0.001,
            verbose=1
            )
res = ZuffyFitIterator(zuffy, fuzzy_X, y, n_iter=20, split_at=0.01, random_state=3112)

perf = res.getPerformance()
best_iter = res.getBestIter()

print(perf)

print(f'bgc: {target_name}={res.getBestClass()}')

feat_imp = visuals.show_feature_importance(
    res.getBestEstimator(),
    fuzzy_X,
    y,
    fuzzy_features_names,
    outputFilename='sample2_impfeat'
    )

visuals.plot_evolution(
    res.getBestEstimator(),
    target_classes,
    res.getPerformance(),
    outputFilename=f'sample2_analysis_{res.getBestScore()*1000:.0f}')

visuals.graphviz_tree(
    res.getBestEstimator(),
    targetFeatureName = target_name,
    #targetClassNames=targetClassNames,    
    featureNames=fuzzy_features_names,
    impFeat=feat_imp,
    treeName=f"spaceshipTitanic Dataset (best accuracy: {res.getBestScore():.3f})\n",
    outputFilename=f'sample2_fpt_{res.getBestScore()*1000:.0f}')

print(metrics.classification_report
      (y, res.getBestEstimator().predict(fuzzy_X)))