import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from zuffy import ZuffyClassifier
from zuffy.zuffy_fit_iterator import ZuffyFitIterator
from zuffy.fuzzy_transformer import convert_to_numeric
# Import all operators, as they are used in the function_set
from zuffy._fpt_operators import (
    COMPLEMENT, CONCENTRATOR, CONCENTRATOR3, CONCENTRATOR4, DIFFUSER,
    DILUTER, DILUTER3, DILUTER4, FUZZY_AND, FUZZY_OR, HAMACHER025, HAMACHER050,
    IFGTE, IFGTE2, IFLT, IFLT2, INTENSIFIER, LUKASIEWICZ_AND,  LUKASIEWICZ_OR, MAXIMUM, MINIMUM,
    PRODUCT, 
    WA_P1, WA_P2, WA_P3, WA_P4, WA_P5, WA_P6, WA_P7, WA_P8, WA_P9, # assuming these are generated in _fpt_operators
    OWA_P1, OWA_P2, OWA_P3, OWA_P4, OWA_P5, OWA_P6, OWA_P7, OWA_P8, OWA_P9, # assuming these are generated in _fpt_operators
)

# Define the full function set as per your original script's first definition
# If you only want the smaller set, comment out or remove the first definition
# and use the second one. I'll use the larger one as it covers more operators.
FULL_FUNCTION_SET = [
    COMPLEMENT,
    CONCENTRATOR,
    CONCENTRATOR3,
    CONCENTRATOR4,
    DIFFUSER,
    DILUTER,
    DILUTER3,
    DILUTER4,
    FUZZY_AND,
    FUZZY_OR,
    HAMACHER025,
    HAMACHER050,
    IFGTE,
    IFGTE2,
    IFLT,
    IFLT2,
    INTENSIFIER,
    LUKASIEWICZ_AND,
    LUKASIEWICZ_OR,
    MAXIMUM,
    MINIMUM,
    PRODUCT,
    WA_P1, WA_P2, WA_P3, WA_P4, WA_P5, WA_P6, WA_P7, WA_P8, WA_P9,
    OWA_P1, OWA_P2, OWA_P3, OWA_P4, OWA_P5, OWA_P6, OWA_P7, OWA_P8, OWA_P9
]

# The script then redefines it to a smaller set.
# Use this if you want to explicitly test with the smaller set that was actually used.
SMALLER_FUNCTION_SET = [
    COMPLEMENT,
    FUZZY_AND,
    FUZZY_OR,
    LUKASIEWICZ_AND,
    LUKASIEWICZ_OR,
    MAXIMUM,
    MINIMUM,
    PRODUCT,
]


@pytest.fixture(scope="session")
def iris_data():
    """Fixture for Iris dataset."""
    iris = load_iris()
    dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    dataset['target'] = iris.target
    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]
    target_class_names = list(iris.target_names)
    return X, y, target_class_names

@pytest.fixture(scope="session")
def pima_data(tmp_path_factory):
    """Fixture for Pima Indian Diabetes dataset.
    Creates a dummy CSV for testing."""
    # Create a dummy CSV file for testing
    dummy_csv_content = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
"""
    # Use tmp_path_factory to create a temporary directory for the dummy CSV
    data_dir = tmp_path_factory.mktemp("datasets")
    pima_path = data_dir / "diabetes.csv"
    with open(pima_path, "w") as f:
        f.write(dummy_csv_content)

    my_data = pd.read_csv(pima_path, sep=',')
    features = my_data.columns
    target_name = str(features[-1])
    # features = features.delete(-1) # Not needed as we use iloc
    X = my_data.iloc[:, 0:-1]
    y = my_data.iloc[:, -1]
    target_names = ['class_0', 'class_1']
    return X, y, target_names

@pytest.fixture(scope="session")
def penguin_data(tmp_path_factory):
    """Fixture for Penguin dataset preprocessing.
    Creates a dummy CSV for testing."""
    dummy_csv_content = """species,island,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g,sex
Adelie,Torgersen,39.1,18.7,181,3750,MALE
Adelie,Torgersen,39.5,17.4,186,3800,FEMALE
Adelie,Biscoe,40.3,18,195,3250,FEMALE
Chinstrap,Dream,46.5,17.9,192,3500,FEMALE
Gentoo,Biscoe,47.3,13.8,195,3900,FEMALE
Adelie,Dream,38.1,17.6,187,3400,FEMALE
Adelie,Dream,37.8,18.3,174,3400,MALE
Adelie,Torgersen,34.1,18.1,193,3475,MALE
Adelie,Torgersen,42.0,20.2,190,4250,MALE
Chinstrap,Dream,49.2,18.2,195,4400,MALE
Gentoo,Biscoe,50.1,15.1,202,4000,FEMALE
"""
    data_dir = tmp_path_factory.mktemp("datasets")
    penguin_path = data_dir / "penguins_size.csv"
    with open(penguin_path, "w") as f:
        f.write(dummy_csv_content)

    dataset_name = 'penguin'
    non_fuzzy_features = ['sex', 'island'] # Renamed from 'non_fuzzy' to avoid conflict with 'non_fuzzy' local var

    my_data = pd.read_csv(penguin_path, sep=',', header=0, skiprows=0)

    # Simplified NaN handling for dummy data
    my_data.dropna(how='any', inplace=True)

    # Drop rows where sex is unknown
    my_data = my_data[my_data['sex'].isin(['MALE','FEMALE'])]

    target_name = 'species'
    target_classes, my_data = convert_to_numeric(my_data, target_name)

    # Now convert island and sex to numeric
    feature_classes = {}
    new_non_fuzzy_cols = [] # Renamed from 'new_non_fuzzy'
    for f in my_data.select_dtypes(exclude=['number']):
        if f in non_fuzzy_features:
            ohe = pd.get_dummies(my_data[f], prefix=f).astype(int) # Add prefix directly
            new_non_fuzzy_cols.extend(ohe.columns)
            my_data = pd.concat([my_data, ohe], axis=1)
            my_data = my_data.drop(f, axis=1)
        else:
            my_data[f], uniques = pd.factorize(my_data[f])
            feature_classes[f] = uniques

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)
    crisp_features = X.columns
    
    return my_data, dataset_name, crisp_features, target_name, list(target_classes), X, y, feature_classes, new_non_fuzzy_cols

@pytest.fixture(scope="function")
def zuffy_classifier_instance():
    """Fixture for a basic ZuffyClassifier instance with minimal settings."""
    # Use the smaller function set for faster tests
    return ZuffyClassifier(generations=2, population_size=10, parsimony_coefficient=0.005, verbose=0, function_set=SMALLER_FUNCTION_SET)

@pytest.fixture(scope="function")
def trained_zuffy_iterator(zuffy_classifier_instance, pima_data):
    """Fixture for a trained ZuffyFitIterator."""
    X, y, _ = pima_data
    # Reduce n_iter and test_size for quicker testing
    res = ZuffyFitIterator(zuffy_classifier_instance, X, y, n_iter=1, test_size=0.5, random_state=103)
    #res.fit() # Manually call fit() as it's not done in __init__
    return res