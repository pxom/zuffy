"""
This module provides a scikit-learn compatible `FuzzyTransformer`
for converting numerical features into fuzzy membership values and
handling categorical features via one-hot encoding.
It also includes helper functions for fuzzy logic operations.

"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

def trimf(feature: np.ndarray, abc: Sequence[float]) -> np.ndarray:
    """
    Calculates the fuzzy membership values of a feature using the triangular
    membership function.

    The triangular membership function is defined by three parameters `[a, b, c]`,
    where 'a' and 'c' are the base points and 'b' is the peak point.
    Membership value is 0 for `feature <= a` or `feature >= c`, 1 for `feature = b`,
    and linearly interpolated between `a` and `b`, and `b` and `c`.

    Parameters
    ----------
    feature : numpy.ndarray
        Crisp input values (e.g., a feature vector). Must be a 1D array or
        convertible to a 1D array.

    abc : Sequence[float], length 3
        Parameters defining the triangular function: `[a, b, c]`.
        Parameters `a` and `c` are the base of the function and `b` is the peak.
        Requires `a <= b <= c`.

    Returns
    -------
    y : numpy.ndarray
        A 1D array of fuzzy membership values represented by the triangular
        membership function, with values in the range `[0, 1]`.

    Raises
    ------
    ValueError
        If `abc` does not have exactly three elements or if `a > b` or `b > c`.
    """
    if len(abc) != 3:
        raise ValueError("`abc` parameter must have exactly three elements [a, b, c].")

    a, b, c = np.asarray(abc, dtype=float)
    if not (a <= b <= c):
        raise ValueError("`abc` parameters must satisfy the condition `a <= b <= c`.")

    feature = np.asarray(feature, dtype=float)
    y = np.zeros_like(feature, dtype=float)

    # Peak of the triangle (where membership is 1.0)
    y[feature == b] = 1.0

    if a != b:
        # Left side of the triangle (rising slope)
        mask = (a < feature) & (feature < b)
        y[mask] = (feature[mask] - a) / (b - a)

    # Right side of the triangle (falling slope)
    if b != c:
        mask = (b < feature) & (feature < c)
        y[mask] = (c - feature[mask]) / (c - b)

    return y

def convert_to_numeric(df: pd.DataFrame, target: str) ->Tuple[List[str], pd.DataFrame]:
    """
    Converts values in a specified target column of a DataFrame into integers
    using `LabelEncoder`.

    This utility function is useful for preparing categorical target variables
    for machine learning models that require numerical inputs.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    target : str
        The name of the column in the DataFrame to be converted.

    Returns
    -------
    classes : numpy.ndarray
        A NumPy array of the original unique class labels found in the target
        column, in the order they were encoded.
    df : pandas.DataFrame
        The DataFrame with the specified target column converted to integer labels.

    Raises
    ------
    ValueError
        If the `target` column is not found in the DataFrame.        
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")
    
    le = LabelEncoder()
    # Apply LabelEncoder to the target column and store original classes
    df[target] = le.fit_transform(df[target])
    return list(le.classes_), df

# pylint: disable=too-many-instance-attributes
class FuzzyTransformer(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible transformer that converts numerical features into
    fuzzy membership values using triangular membership functions. It also
    handles specified non-fuzzy (categorical) columns using one-hot encoding.

    This transformer is designed to preprocess data for Fuzzy Pattern Tree
    models by converting raw numerical inputs into fuzzy memberships and
    handling nominal categorical features.

    Parameters
    ----------
    non_fuzzy : list of str, default=None
        A list of column names that should NOT be fuzzified. These columns will
        instead be one-hot encoded. If `None` or an empty list, all detected
        numerical columns will be fuzzified.

    oob_check : bool, default=False
        If `True`, the transformer will check for out-of-bounds numerical values
        (values falling outside the min/max range observed during `fit` for
        fuzzified columns) and raise a `ValueError`.
        If `False` (default), such out-of-bounds values will have their fuzzy
        membership values automatically clamped to 0.0 by the `trimf` function
        without raising an error.

    tags : list of str, default=['low', 'med', 'high']
        A list of three strings to be used as suffixes for the generated fuzzy
        feature names (e.g., 'low_feature_name', 'med_feature_name').
        The order corresponds to the 'low', 'medium', and 'high' membership
        functions, respectively.  Must contain exactly three strings.

    feature_names : list of str, default=None
        Optional list of input feature names.
        - If `X` passed to `fit` is a Pandas DataFrame, its column names will be
          used, and this parameter will be ignored.
        - If `X` is a NumPy array, these names will be used to assign column names
          for internal DataFrame processing and output feature naming. The length
          must match `X.shape[1]`.

    show_fuzzy_range : bool, default=True
        If `True`, the names of the output fuzzy features will include the
        numerical range (e.g., 'low feature_name (0.00 to 5.00)').
        If `False`, only the tag and feature name will be used (e.g., 'low feature_name').

    verbose : bool, default=False
        If `True`, the transformer will print progress and debugging
        information during `fit` and `transform` operations.

    Attributes
    ----------
    n_features_in\\_ : int
        The number of features seen during :term:`fit`.

    feature_names_in\\_ : ndarray of str or None
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings (e.g., a Pandas DataFrame), or
        when `feature_names` is explicitly provided in `__init__` and `X` is a
        NumPy array.

    columns\\_ : pandas.Index
        The column names of the input DataFrame (or names derived from
        `feature_names` if a NumPy array was provided) seen during `fit`.
        This attribute ensures consistency between `fit` and `transform` calls.

    fuzzy_bounds\\_ : dict
        A dictionary storing the `(a, b, c)` parameters for the triangular
        membership functions for each numerical column that was fuzzified.
        Keys are original column names, values are tuples of floats.

    categorical_values\\_ : dict
        A dictionary storing the unique categories for each column specified in
        `non_fuzzy`. This ensures consistent one-hot encoding during `transform`.
        Keys are original column names, values are lists of unique categories.

    feature_names_out\\_ : list of str
        A list of names for the features generated after transformation. This
        attribute is available after the `fit` method has been called.

    See Also
    --------
    trimf : Triangular membership function.
    pandas.get_dummies : Convert categorical variable into dummy/indicator variables.

    Notes
    -----
    The transformer automatically identifies numerical columns for fuzzification.
    Columns specified in `non_fuzzy` (which are assumed categorical) are handled
    via one-hot encoding. If `X` is a NumPy array, it expects numerical data
    only unless `non_fuzzy` is used and `feature_names` is explicitly provided.
    """

    _parameter_constraints: Dict[str, List[Any]] = {
        "non_fuzzy": [list, None],
        "oob_check": [bool],
        "tags": [list],
        "feature_names": [list, None],
        "show_fuzzy_range": [bool],
        "verbose": [bool],
    }

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        non_fuzzy: Optional[List[str]] = None,
        oob_check: bool = False,
        tags: List[str] = ['low', 'med', 'high'],
        feature_names: Optional[List[str]] = None,
        show_fuzzy_range: bool = True,
        verbose: bool = False,
    ):
        self.non_fuzzy = non_fuzzy if non_fuzzy is not None else []
        self.oob_check = oob_check
        self.tags = tags
        self.feature_names = feature_names
        self.show_fuzzy_range = show_fuzzy_range
        self.verbose = verbose

        # Validate tags length and type early for robustness.
        if not isinstance(self.tags, (list, tuple)):
            raise TypeError("`tags` must be a list or tuple.")
        if len(self.tags) != 3:
            raise ValueError(f"`tags` must contain exactly three strings, but found {len(self.tags)}.")
        if not all(isinstance(t, str) for t in self.tags):
            raise TypeError("All elements in `tags` must be strings.")


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) \
            -> "FuzzyTransformer":
        """Fits the FuzzyTransformer by determining fuzzy bounds for numerical features
        and unique categories for categorical features.

        This method analyzes the input data `X` to learn the necessary parameters
        for subsequent transformation.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
            The training input samples. If a NumPy array, it will be internally
            converted to a DataFrame to handle columns by name, using `feature_names`
            provided in `__init__` or default names.

        y : None, default=None
            Ignored. Present for API consistency as a scikit-learn transformer.

        Returns
        -------
        self : FuzzyTransformer
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If `X` is a NumPy array and `feature_names` was not provided in `__init__`.
            If a column designated for fuzzification contains non-numeric data.
        """
        if not isinstance(X, pd.DataFrame) and self.feature_names is None:
            raise ValueError(
                "You must specify `feature_names` in the `__init__` method "
                "if you use NumPy arrays in the `fit` method."
            )

        # Determine column count based on input type
        if isinstance(X, pd.DataFrame):
            col_cnt = X.shape[1]
        else:
            if X.ndim == 1:  # Handle 1D array case if it means single feature
                col_cnt = 1
            else:
                col_cnt = X.shape[1]

        # Validate feature_names length against column count
        if self.feature_names is not None and len(self.feature_names) != col_cnt:
            raise ValueError(f"The number of feature_names supplied ({len(self.feature_names)}) " \
                             f"does not match the number of columns in X ({len(X[0])})")

        # Validate X and set n_features_in_ and feature_names_in_
        _ = self._validate_data(
            X.copy(),
            accept_sparse=False,
            force_all_finite='allow-nan',
            reset=True,
            dtype=object
        )

        # Convert to DataFrame if not already, to handle columns by name.
        # Prioritize names from the fitted `feature_names_in_` or the
        # `feature_names` parameter from __init__ if X was a NumPy array.
        if not isinstance(X, pd.DataFrame):
            # If X was a NumPy array, `self.feature_names_in_` would be None after `_validate_data`.
            # In that case, use the names provided in `__init__` (if any), or default names.
            columns_for_df = self.feature_names
            if columns_for_df is None and self.feature_names is not None:
                columns_for_df = self.feature_names
            elif columns_for_df is None:
                columns_for_df = [f"x{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=columns_for_df)
        else:
            X_df = X.copy() # Work on a copy to avoid modifying original DataFrame

        self.columns_ = X_df.columns
        # Store (min, mid, max) for numeric columns
        self.fuzzy_bounds_: Dict[str, Tuple[float, float, float]] = {}
        # Store categories for one-hot consistency
        self.categorical_values_: Dict[str, List[Any]] = {}
        self.feature_names_out_: List[str] = [] # Initialize output feature names

        for col in self.columns_:
            if col in self.non_fuzzy:
                # Handle categorical columns by recording unique values for one-hot encoding
                if not X_df[col].empty:
                    self.categorical_values_[col] = sorted(X_df[col].dropna().unique().tolist())
                else:
                    self.categorical_values_[col] = [] # Handle empty column gracefully

                # Add names for one-hot encoded features (col=category)
                for cat in self.categorical_values_[col]:
                    self.feature_names_out_.append(f"{col}= {cat}")
            else:
                # Handle fuzzy (numerical) columns
                values = X_df[col].dropna().values
                if not np.issubdtype(values.dtype, np.number):
                    raise ValueError(f"Column '{col}' must contain numeric data for fuzzification, "
                                     f"but found non-numeric type: {values.dtype}. "
                                     f"Consider adding it to `non_fuzzy` or converting its dtype.")

                if values.size == 0:
                    # If column is empty, set default bounds (e.g., for consistency)
                    a, b, c = 0.0, 0.0, 0.0
                else:
                    a = float(np.min(values))
                    c = float(np.max(values))
                    b = a + (c - a) / 2.0 # Mid-point of the range
                self.fuzzy_bounds_[col] = (a, b, c)

                # Add names for the three fuzzy features (low, med, high)
                if self.show_fuzzy_range:
                    self.feature_names_out_.extend([
                        f"{self.tags[0]} {col}| ({a:.2f} to {b:.2f})",
                        f"{self.tags[1]} {col}| ({a:.2f} to {b:.2f} to {c:.2f})",
                        f"{self.tags[2]} {col}| ({b:.2f} to {c:.2f})"
                    ])
                else:
                    self.feature_names_out_.extend([
                        f"{self.tags[0]} {col}",
                        f"{self.tags[1]} {col}",
                        f"{self.tags[2]} {col}"
                    ])

                if self.verbose:
                    print(f"Fitted fuzzy bounds for '{col}': a={a:.2f}, b={b:.2f}, c={c:.2f}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms the input data into fuzzy membership values and one-hot
        encoded categorical features based on the `fit` method's learnings.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
            The input data to transform. It must have the same number of features
            and, if a DataFrame, the same column names and order as the data used
            during `fit`.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_transformed_features)
            The transformed data, consisting of fuzzy membership values for numerical
            columns and one-hot encoded values for categorical columns.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet (i.e., `fit` has not been called).
        ValueError
            If the number of features in `X` does not match `n_features_in_`,
            if DataFrame column names/order mismatch `columns_`, or if out-of-bounds
            numerical values are encountered and `oob_check` is `True`.
        """
        check_is_fitted(self) # Ensure the transformer has been fitted.

        # Convert to DataFrame if not already, using the column names seen during fit.
        if not isinstance(X, pd.DataFrame):
            # _validate_data already checked n_features.
            X_df = pd.DataFrame(X, columns=self.columns_)
        else:
            X_df = X.copy()

        # Zeroise non fuzzy columns
        temp_X = X_df.copy()
        temp_X[self.non_fuzzy] = 0

        # Validate X and set n_features_in_ and feature_names_in_
        # Validate input `X` for transformation. `reset=False` ensures that
        # `n_features_in_` is checked against the fitted value.
        _ = self._validate_data(
            temp_X,
            reset=False,
            accept_sparse=False,
            force_all_finite='allow-nan'
        )

        # Explicitly check if DataFrame columns match the fitted columns.
        if not self.columns_.equals(X_df.columns):
            raise ValueError(
                "The columns in the input DataFrame for `transform` do not match "
                "the columns seen during `fit`. Please ensure the same column "
                "names and order."
            )

        transformed_features: List[np.ndarray] = []

        for col in self.columns_:
            values = X_df[col].values
            if col in self.non_fuzzy:
                # One-hot encode categorical features.
                # `pd.get_dummies` with `prefix`/`prefix_sep` handles naming.
                one_hot = pd.get_dummies(values, prefix=col, prefix_sep='= ').astype(int)

                # Ensure all categories seen during fit are present in the output,
                # adding zero columns for any missing categories (e.g., unseen in test set).
                expected_cols = [f"{col}= {cat}" for cat in self.categorical_values_[col]]
                for expected_col in expected_cols:
                    if expected_col in one_hot.columns:
                        transformed_features.append(one_hot[expected_col].values.reshape(-1, 1))
                    else:
                        # If a category is missing in the current `X` (e.g., not in test set),
                        # add a column of zeros to maintain consistent output shape.
                        transformed_features.append(np.zeros((X.shape[0], 1), dtype=int))
            else:
                # Apply fuzzy transformation using learned bounds.
                a, b, c = self.fuzzy_bounds_[col]

                # Check for out-of-bounds values if oob_check is enabled.
                if self.oob_check:
                    if np.any(values < a):
                        out_of_bounds_vals = values[values < a]
                        raise ValueError(
                            f"The '{col}' feature has values "
                            f"({np.array2string(out_of_bounds_vals, max_line_width=100)}) "
                            f"that are less than 'a' ({a:.2f}). Set `oob_check=False` to "
                            f"ignore this warning."
                        )
                    if np.any(values > c):
                        out_of_bounds_vals = values[values > c]
                        raise ValueError(
                            f"The '{col}' feature has values "
                            f"({np.array2string(out_of_bounds_vals, max_line_width=100)}) "
                            f"that are greater than 'c' ({c:.2f}). Set `oob_check=False` to "
                            f"ignore this warning."
                        )

                # Calculate fuzzy membership values for low, medium, and high bands.
                # 'low': ramps from 1 at domain_min to 0 at 'b' (midpoint)
                lo = trimf(values, [a, a, b])
                # 'medium': standard triangle from 'a' to 'c' with peak at 'b'
                md = trimf(values, [a, b, c])
                # 'high': ramps from 0 at 'b' (midpoint) to 1 at domain_max
                hi = trimf(values, [b, c, c])
                transformed_features.append(np.column_stack([lo, md, hi]))

        # Concatenate all transformed features horizontally to form the final output array.
        return np.hstack(transformed_features)
