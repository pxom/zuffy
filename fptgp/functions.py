import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#print('*** in functions.py - included by __init__***')

# from generatemf.py of skfuzzy
def trimf(x, abc):
    """
    Triangular membership function generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    abc : 1d array, length 3
        Three-element vector controlling shape of triangular function.
        Requires a <= b <= c.

    Returns
    -------
    y : 1d array
        Triangular membership function.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

def fuzzify_col(col: np.array, feature_name: str, info: bool = False, tags: list[str] = None) -> list[float] | str: # Three bands of fuzzification
    min = np.min(col)
    max = np.max(col)
    mid = int((max-min)/2) + min

    # return three new features
    # min -> mid
    # min -> max
    # med -> max
    #print(f'fuzzify_col: {min} < {mid} < {max}')

    lo = trimf(col, [min, min, mid])
    md = trimf(col, [min, mid, max])
    hi = trimf(col, [mid, max, max])

    mid = round(mid,2) # because of fp inaccuracy giving ...9999999998 etc

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

def fuzzify_data(data: pd.DataFrame, non_fuzzy: list = [], info: bool = False, tags: list[str] = None):
    if type(data) != pd.DataFrame:
        raise ValueError("The 'data' parameter is not a valid Pandas DataFrame")
    fuzzy_X = None
    #for col, feature_name in zip(data.columns, crisp_features):
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
                res = pd.get_dummies(data[feature_name],prefix=feature_name)
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

def convert_to_numeric(my_data: pd.DataFrame, target):
    le = LabelEncoder()
    my_data[target] = le.fit_transform(my_data[target])
    target_classes = le.classes_
    #for colname, type in zip(my_data.columns,my_data.dtypes):
    #    print(f"{colname} is {type.name}")
    return target_classes, my_data

