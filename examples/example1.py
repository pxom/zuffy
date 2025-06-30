'''
Zuffy Example 1 - Minimal
'''

# pylint: disable=no-member
import pandas as pd

from sklearn.datasets import load_iris

from zuffy import ZuffyClassifier, visuals
from zuffy.zuffy_fit_iterator import ZuffyFitIterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

zuffy = ZuffyClassifier(generations=15)
fit_iterator = ZuffyFitIterator(zuffy, n_iter=3, show_fuzzy_range=False)
trained_iterator_results = fit_iterator.fit(X, y)

visuals.graphviz_tree(
    trained_iterator_results.best_estimator_,
    target_class_names=list(iris.target_names),
    target_feature_name = 'Species',
    feature_names=trained_iterator_results.fuzzy_feature_names_,
    tree_name=f"Iris Dataset (Accuracy: {trained_iterator_results.best_score_:.3f})",
    output_filename=f'example1_fpt_{trained_iterator_results.best_score_*1000:.0f}')
