"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
Functions to handle the display of a FPT
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Union, Dict, Any, Sequence

def trimf(feature: np.ndarray, abc: Sequence[float]) -> np.ndarray:
    """
    This calculates the fuzzy membership values of a feature using the triangular membership function.

    The triangular membership function is defined by three parameters [a, b, c],
    where 'a' and 'c' are the base points and 'b' is the peak point.
    Membership value is 0 for feature <= a or feature >= c, 1 for feature = b, and linearly
    interpolated between a and b, and b and c.    

    Parameters
    ----------
    feature : numpy.ndarray
        Crisp input values (e.g., a feature vector). Must be a 1D array.

    abc : Sequence[float], length 3
        Parameters defining the triangular function: [a, b, c].
        Parameters a and c are the base of the function and b is the peak.
        Requires `a <= b <= c`.

    Returns
    -------
    y : numpy.ndarray
        A 1D array of fuzzy membership values representing by the triangular membership function.

    Raises
    ----------
    ValueError
        If `abc` does not have exactly three elements or if `a > b` or `b > c`.
    """
    if len(abc) != 3:
        raise ValueError("`abc` parameter must have exactly three elements [a, b, c].")
    
    a, b, c = np.asarray(abc, dtype=float)
    if not (a <= b <= c):
        raise ValueError("`abc` parameters must satisfy the condition `a <= b <= c`.")

    y = np.zeros_like(feature, dtype=float)

    # Left side
    if a != b:
        mask = (a < feature) & (feature < b)
        y[mask] = (feature[mask] - a) / (b - a)

    # Right side
    if b != c:
        mask = (b < feature) & (feature < c)
        y[mask] = (c - feature[mask]) / (c - b)

    y[feature == b] = 1.0
    return y


def fuzzify_col(col: np.array, feature_name: str, info: bool = False, tags: list[str] = None) -> list[float] | str:
    """
    Fuzzifies a numeric column into three overlapping membership functions.

    Parameters
    ----------
    col : np.ndarray
        1D array of numeric values.
    feature_name : str
        Name of the feature (column).
    info : bool
        Whether to print debug information.
    tags : List[str], optional
        Optional names for the fuzzy bands (e.g., ['low', 'med', 'high']).

    Returns
    -------
    Tuple[List[np.ndarray], Union[List[str], None]]
        Three fuzzified arrays and their names.
    """
    if not is_numeric_dtype(col):
        raise ValueError(f"This column ({feature_name}) must contain numeric data but it does not.")
    
    if not np.issubdtype(col.dtype, np.number):
        raise ValueError(f"np This column ({feature_name}) must contain numeric data but it does not.")
    
    min = np.min(col)
    max = np.max(col)
    mid = int((max-min)/2.0) + min

    # return three new features
    # min -> mid
    # min -> max
    # med -> max

    lo = trimf(col, [min, min, mid])
    md = trimf(col, [min, mid, max])
    hi = trimf(col, [mid, max, max])

    mid = round(mid,2) # for display because of fp inaccuracy giving ...9999999998 etc

    if tags:
        new_feature_names = []
        new_feature_names.append(str(tags[0]) + ' ' + str(feature_name) + f"| ({min} to {mid})")
        new_feature_names.append(str(tags[1]) + ' ' + str(feature_name) + f"| ({min} to {mid} to {max})")
        new_feature_names.append(str(tags[2]) + ' ' + str(feature_name) + f"| ({mid} to {max})")
    else:
        new_feature_names = None
    # turn each into a named feature in a np array
    if info:
        print(f"{feature_name} => {min} < {mid} < {max} ", end = ' ')
    return [lo,md,hi], new_feature_names


def fuzzify_data(data: pd.DataFrame, non_fuzzy: list = [], info: bool = False, tags: list[str] = ['low', 'med', 'high']):
    if type(data) != pd.DataFrame:
        raise ValueError("The 'data' parameter is not a valid Pandas DataFrame")
    
    fuzzy_X = None
    fuzzy_feature_names = []
    for feature_name in data.columns:
        #res = np.transpose(fuzzify_col(np.array(data[col]), feature_name, info=False))
        if feature_name not in non_fuzzy:
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # NEED TO WORK HERE!
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            fuzzy_range, new_feature_names = fuzzify_col(np.array(data[feature_name]), feature_name, info=info, tags=tags)
            fuzzy_feature_names.append(new_feature_names)
            res = np.transpose(fuzzy_range)
        else:
            # TODO: 250104
            # feature is non fuzzy but already converted to numeric so we should one hot encode it with 
            # the label matching the original value
            # e.g. cap-size=G, cap-size=R
            if 1:
                res = pd.get_dummies(data[feature_name],prefix=f"{feature_name}",prefix_sep='= ')
                res = res.astype(int) # convert True/False to 1/0
                fuzzy_feature_names.append(list(res.columns))
            else:
                res = data[feature_name]
                res = np.reshape(res,(len(res),1))
                fuzzy_feature_names.append([feature_name])
        # append res to our matrix
        if isinstance(fuzzy_X, np.ndarray) or isinstance(fuzzy_X, pd.DataFrame):
            fuzzy_X = np.concatenate((fuzzy_X, res),axis=1)
        else:
            fuzzy_X = res
    if tags:
        fuzzy_feature_names = flatten(fuzzy_feature_names)
    return fuzzy_X, fuzzy_feature_names


def flatten(matrix: list[list[float]]) -> list[float]:
    '''
    Flatten a list of arrays into a single dimensional list of values using concatenation.
    '''
    flat_list = []
    for row in matrix:
        if isinstance(row, (list, tuple, np.ndarray)):
            flat_list += row
        else:
            raise ValueError(f"I cannot flatten a list that does not contain lists (found '{row}' in the matrix and I expected a list)")
    return flat_list


def fuzzy_feature_names(flist: list[str], tags: list[str]) -> list[str]:
    '''
    Generate a list of fuzzy feature names which have each of the tags appended.
    '''
    if not isinstance(flist, (list, tuple, np.ndarray)):
        raise ValueError(f"fuzzy_feature_names expects first parameter flist to be a list")
    
    new_features = []
    for f in flist:
            if not isinstance(f, str):
                raise ValueError(f"fuzzy_feature_names expects first parameter, flist, to be a list of strings but found ",f,type(f))
            for tag in tags:
                if not isinstance(tag, str):
                    raise ValueError(f"fuzzy_feature_names expects the second parameter, tags, to be a list of stringsbut found ",tag,type(tag))
                new_features.append(tag + ' ' + f)
    return new_features


def convert_to_numeric(df: pd.DataFrame, target):
    '''
    This converts the values in the target column into integers and
    returns a list of the original values prior to conversion.
    '''
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    #target_classes = le.classes_
    #for colname, type in zip(my_data.columns,my_data.dtypes):
    #    print(f"{colname} is {type.name}")
    return le.classes_, df


def convert_to_numeric2(df: pd.DataFrame): # single column
    '''
    This converts the values in the target column into integers and
    returns a list of the original values prior to conversion.
    '''
    le = LabelEncoder()
    df = le.fit_transform(df)
    #target_classes = le.classes_
    #for colname, type in zip(my_data.columns,my_data.dtypes):
    #    print(f"{colname} is {type.name}")
    return le.classes_, df


class xxxFuzzyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, non_fuzzy: list[str] = None, tags: list[str] = ['low', 'med', 'high'], info: bool = False):
        self.non_fuzzy = non_fuzzy or []
        self.tags = tags
        self.info = info
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.columns_ = X.columns
        return self

class FuzzyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, non_fuzzy: list[str] = None, tags: list[str] = ['low', 'med', 'high'], info: bool = False):
        self.non_fuzzy = non_fuzzy or []
        self.tags = tags
        self.info = info

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        self.columns_ = X.columns
        self.fuzzy_bounds_ = {}  # Store (min, mid, max) for numeric columns
        self.categorical_values_ = {}  # Store categories for one-hot consistency
        self.feature_names_ = []

        for col in self.columns_:
            if col in self.non_fuzzy:
                self.categorical_values_[col] = sorted(X[col].dropna().unique().tolist())
            else:
                values = X[col].dropna().values
                if not is_numeric_dtype(values):
                    raise ValueError(f"Column '{col}' must be numeric to be fuzzified.")
                a = float(np.min(values))
                c = float(np.max(values))
                b = a + (c - a) / 2.0
                self.fuzzy_bounds_[col] = (a, b, c)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fuzzy_bounds_"):
            raise RuntimeError("You must fit the transformer before transforming data.")

        fuzzy_X = []
        self.feature_names_ = []

        for col in self.columns_:
            if col in self.non_fuzzy:
                categories = self.categorical_values_[col]
                one_hot = pd.get_dummies(X[col], prefix=col, prefix_sep='= ')
                for cat in categories:
                    name = f"{col}= {cat}"
                    if name in one_hot.columns:
                        col_data = one_hot[name].values.reshape(-1, 1)
                    else:
                        # unseen category during fit
                        col_data = np.zeros((X.shape[0], 1))
                    fuzzy_X.append(col_data)
                    self.feature_names_.append(name)
            else:
                if col not in self.fuzzy_bounds_:
                    raise ValueError(f"No fuzzy bounds found for column '{col}' — did you fit this transformer?")
                a, b, c = self.fuzzy_bounds_[col]
                values = X[col].values

                if np.any(values < a):
                    raise ValueError(f"The '{col}' feature has values ({values[values<a]}) that are less than 'a' ({a}) so it cannot be fuzzified")

                if np.any(values > c):
                    raise ValueError(f"The '{col}' feature has values ({values[values>c]}) that are greater than 'c' ({c}) so it cannot be fuzzified")

                lo = trimf(values, [a, a, b])
                md = trimf(values, [a, b, c])
                hi = trimf(values, [b, c, c])

                fuzzy_X.append(np.column_stack([lo, md, hi]))
                self.feature_names_.extend([
                    f"{self.tags[0]} {col}| ({a} to {b})",
                    f"{self.tags[1]} {col}| ({a} to {b} to {c})",
                    f"{self.tags[2]} {col}| ({b} to {c})"
                ])

                if self.info:
                    print(f"{col} => {a:.2f} < {b:.2f} < {c:.2f}")

        return np.hstack(fuzzy_X)

    def get_feature_names_out(self) -> list[str]:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Transformer has not been fitted or transformed yet.")
        return self.feature_names_