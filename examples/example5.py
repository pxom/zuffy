'''
Zuffy Example 5 - Full Visuals
'''

# pylint: disable=no-member
import pandas as pd

from sklearn.datasets import load_iris

from zuffy import ZuffyClassifier
from zuffy.visuals import (
                        show_feature_importance,
                        plot_evolution,
                        graphviz_tree,
                        plot_iteration_performance
                        )
from zuffy.zuffy_fit_iterator import ZuffyFitIterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

zuffy = ZuffyClassifier(generations=15, parsimony_coefficient=0.0002)
fit_iterator = ZuffyFitIterator(zuffy, n_iter=10, show_fuzzy_range=True)
trained_iterator_results = fit_iterator.fit(X, y)

best_zuffy_model = trained_iterator_results.best_estimator_
best_overall_score = trained_iterator_results.best_score_
fuzzy_feature_names = trained_iterator_results.fuzzy_feature_names_
iter_perf = trained_iterator_results.iteration_performance_

feature_importance_results = show_feature_importance(
    reg=best_zuffy_model,
    X_test=best_zuffy_model.X_,
    y_test=best_zuffy_model.y_,
    features=fuzzy_feature_names, # Use fuzzified feature names for this plot
    n_jobs=1,                     # Number of parallel jobs (-1 for all processors)
    n_repeats=5,                  # Number of permutations
    output_filename='example5_feat_imp'
)

# Plot Evolutionary Metrics (e.g., fitness and length over generations)
plot_evolution(
    model=best_zuffy_model,
    target_class_names=list(iris.target_names),
    output_filename='example5_evolution'
)

graphviz_tree(
    trained_iterator_results.best_estimator_,
    imp_feat=feature_importance_results,
    target_class_names=list(iris.target_names),
    target_feature_name = 'Species',
    feature_names=fuzzy_feature_names,
    tree_name=f"Iris Dataset (Accuracy: {trained_iterator_results.best_score_:.3f})",
    output_filename='example5_fpt')

plot_iteration_performance(
    iter_perf,
    title='Iteration Performance: Example5',
    output_filename='example5_iter_perf'
    )
