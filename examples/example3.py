'''
Zuffy Example 3 - GridSearchCV
'''

# pylint: disable=no-member
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

from zuffy import ZuffyClassifier
from zuffy.zuffy_fit_iterator import ZuffyFitIterator


# 1. Load data
iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

# 2. Define the base estimator and the parameter grid for GridSearchCV
zuffy_model = ZuffyClassifier()

param_grid = {
    'population_size': [5, 20, 500],
    'generations': [3, 10, 20],
}

# 3. Set up GridSearchCV
# This will search for the best hyperparameters for the ZuffyClassifier
# within each iteration of the ZuffyFitIterator.
grid_search = GridSearchCV(
    estimator=zuffy_model,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # Use all available CPU cores,
    verbose=1 # Set to 1 to see detailed output
)

# 4. Initialize and run ZuffyFitIterator
# It will run GridSearchCV 3 times (n_iter=3) and find the best overall model.
fit_iterator = ZuffyFitIterator(
    model=grid_search,
    n_iter=5,
    test_size=0.25
)

trained_iterator_results = fit_iterator.fit(X, y, feature_names=None)

# 5. Display the results
print(f"Best iteration index: {fit_iterator.best_iteration_index_}")
print(f"Best score (accuracy): {fit_iterator.best_score_:.4f}")
print(f"Smallest tree size of best model: {fit_iterator.smallest_tree_size_}")

# Display the best parameters found by GridSearchCV in the best iteration
if hasattr(fit_iterator.best_estimator_, 'best_params_'):
    print(f"Best hyperparameters found by GridSearch: {fit_iterator.best_estimator_.best_params_}")

# Show performance of each iteration
print("\n--- Iteration Performance ---")
print("\n   Score     Tree Size  Class Scores")
perf_df = pd.DataFrame(fit_iterator.iteration_performance_)
print(perf_df.to_string())

# Example of using the fitted iterator to make predictions on new data
print("\n--- Making a prediction on a 'new' data sample ---")
# Take the last 5 samples as "new" data
X_new = X[-5:]
fuzzy_X_new = fit_iterator.fuzz_transformer_.transform(X_new)
predictions = fit_iterator.best_estimator_.predict(fuzzy_X_new)
print(f"Predicted classes: {predictions}")
print(f"Actual classes: {list(y[-5:])}")
