'''
Zuffy Example 2 - Categoricals
'''

# pylint: disable=no-member disable=C0103 # uppercase naming style
import pandas as pd

from zuffy import ZuffyClassifier, visuals
from zuffy.zuffy_fit_iterator import ZuffyFitIterator
from zuffy.fuzzy_transformer import convert_to_numeric

DATASET_FOLDER = '../../datasets/'
my_data = pd.read_csv(DATASET_FOLDER + 'adult.data', sep=',', header=None, skiprows=0)
my_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
                   'occupation','relationship','race','sex','capital-gain','capital-loss',
                   'hours-per-week','native-country','income']

target_name = 'income'
target_class_names, my_data = convert_to_numeric(my_data, target_name)

non_fuzzy =  ['workclass','education','marital-status','occupation',
              'relationship','race','sex','native-country']

y = my_data[target_name]
X = my_data.drop(target_name, axis=1)

zuffy = ZuffyClassifier(generations=25, parsimony_coefficient=0.00002, verbose=1)
fit_iterator = ZuffyFitIterator(zuffy,n_iter=5)
trained_iterator_results = fit_iterator.fit(X, y, non_fuzzy=non_fuzzy)

visuals.plot_evolution(
    trained_iterator_results.best_estimator_,
    target_class_names=target_class_names,
    output_filename='example2_evolution')

visuals.graphviz_tree(
    trained_iterator_results.best_estimator_,
    target_class_names=target_class_names,
    target_feature_name = target_name,
    feature_names=trained_iterator_results.fuzzy_feature_names_,
    tree_name=f"Adult Dataset (Accuracy: {trained_iterator_results.best_score_:.3f})",
    source_filename=f'example2_fpt_{trained_iterator_results.best_score_*1000:.0f}.dot',
    output_filename=f'example2_fpt_{trained_iterator_results.best_score_*1000:.0f}')
