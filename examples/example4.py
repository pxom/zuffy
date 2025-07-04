'''
Zuffy Example 4 - Pipeline
'''

# pylint: disable=no-member
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from zuffy import ZuffyClassifier
from zuffy.fuzzy_transformer import FuzzyTransformer
from zuffy.visuals import plot_evolution


# 1. Create a synthetic dataset
N_SAMPLES = 100

data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=N_SAMPLES),
    'income': np.random.randint(20000, 100000, size=N_SAMPLES),
    'gender': np.random.choice([0, 1], size=N_SAMPLES),
    'region': np.random.choice([1, 2, 3, 4], size=N_SAMPLES),
    'target': np.random.choice(['A', 'B'], size=N_SAMPLES)
})

X = data.drop('target', axis=1)
y = data['target']

# 2. Define preprocessing pipelines for numeric and categorical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Create a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['age','income'])
    ],
    remainder='passthrough'
)

# 3. Create the full pipeline including preprocessing, fuzzification, and classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('fuzzifier', FuzzyTransformer(feature_names=X.columns)),
    ('classifier', ZuffyClassifier(generations=35))
])

# 4. Fit the model
pipeline_results = full_pipeline.fit(X, y)

# 5. Display the results
print(f"Fuzzy Feature Names: {pipeline_results.named_steps['fuzzifier'].feature_names_out_}")

plot_evolution(
    model=pipeline_results.named_steps['classifier'],
    target_class_names=['A', 'B'],
    output_filename='example4_evolution'
    )

print("Run Details:")
print(pipeline_results.named_steps['classifier'].multi_.estimators_[0].run_details_)
