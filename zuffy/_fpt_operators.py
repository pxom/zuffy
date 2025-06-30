"""
This module contains a set of fuzzy logic operators designed for use with Zuffy,
suitable for constructing Fuzzy Pattern Trees (FPT). Each operator performs
a specific mathematical operation on NumPy arrays, representing fuzzy set memberships.

The functions can be organised thus:
    1. Basic Fuzzy Operators
        * _minimum (MINIMUM/and)
        * _maximum (MAXIMUM/or)
        * _complement (COMPLEMENT/not)

    2. Linguistic Hedges
        * _diluter (DILUTER)
        * _diluter_power (used by DILUTER3, DILUTER4)
        * _concentrator (CONCENTRATOR)
        * _concentrator_power (used by CONCENTRATOR3, CONCENTRATOR4)
        * _intensifier (INTENSIFIER)
        * _diffuser (DIFFUSER)

    3. Averaging Operators
        * _weighted_average (used by WA_P1 to WA_P9)
        * _ordered_weighted_average (used by OWA_P1 to OWA_P9)

    4. T-Norms and T-Conorms
        * _fuzzy_and (FUZZY_AND - specifically product t-norm)
        * _fuzzy_or (FUZZY_OR - specifically probabilistic sum t-conorm)
        * _lukasiewicz_t_norm (LUKASIEWICZ/AND)
        * _lukasiewicz_t_conorm (LUKASIEWICZ/OR)
        * _hamacher_t_norm (used by HAMACHER025, HAMACHER050)
        * _product_t_norm (PRODUCT)

    5. Conditional Operators
        * _if_gte (IFGTE)
        * _if_gte_else (IFGTE2)
        * _if_lt (IFLT)
        * _if_lt_else (IFLT2)

"""


from typing import Union

import numpy as np
from gplearn import functions

# Define type alias for clarity
ArrayLike = Union[np.ndarray, float]

def _weighted_average(a: ArrayLike, b: ArrayLike, x: float) -> ArrayLike:
    """
    Calculates the Weighted Average: x*a + (1-x)*b.

    Parameters
    ----------
    a : np.ndarray or float
        The first operand, typically a fuzzy set membership value or array.
    b : np.ndarray or float
        The second operand, typically a fuzzy set membership value or array.
    x : float
        The weight to apply to 'a', with (1-x) applied to 'b'.
        Should be in the range [0, 1] for typical fuzzy operations.

    Returns
    -------
    np.ndarray or float
        The result of the weighted average operation.
    """
    # Ensure x is a float for consistent calculations
    x = float(x)
    return x * a + (1 - x) * b

def _ordered_weighted_average(a: ArrayLike, b: ArrayLike, x: float) -> ArrayLike:
    """
    Calculates the Ordered Weighted Average (OWA): x*max(a, b) + (1-x)*min(a, b).

    Parameters
    ----------
    a : np.ndarray or float
        The first operand.
    b : np.ndarray or float
        The second operand.
    x : float
        The weight to apply to the maximum of 'a' and 'b', with (1-x) applied to the minimum.
        Should be in the range [0, 1].

    Returns
    -------
    np.ndarray or float
        The result of the OWA operation.
    """
    x = float(x)
    return x * np.maximum(a, b) + (1 - x) * np.minimum(a, b)

def _minimum(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """
    Performs the Minimum operation, equivalent to a boolean AND in fuzzy sets.

    Parameters
    ----------
    x0 : np.ndarray or float
        The first operand.
    x1 : np.ndarray or float
        The second operand.

    Returns
    -------
    np.ndarray or float
        The element-wise minimum of x0 and x1.
    """
    return np.minimum(x0, x1)

def _maximum(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """
    Performs the Maximum operation, equivalent to a boolean OR in fuzzy sets.

    Parameters
    ----------
    x0 : np.ndarray or float
        The first operand.
    x1 : np.ndarray or float
        The second operand.

    Returns
    -------
    np.ndarray or float
        The element-wise maximum of x0 and x1.
    """
    return np.maximum(x0, x1)

def _diluter(x0: ArrayLike) -> ArrayLike:
    """
    Applies a Diluter operation (square root) to fuzzy set memberships.
    Typically used to "fuzzify" or expand the meaning of a fuzzy set.
    Ensures non-negative output for non-negative input.

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the dilution (square root) operation.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure that negative values don't result in NaNs from sqrt
        return np.where(x0 < 0, 0, x0**0.5)

def _diluter_power(x0: ArrayLike, power: float) -> ArrayLike:
    """
    Applies a generalized Diluter operation (x0^power) to fuzzy set memberships.
    Ensures non-negative output for non-negative input.

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.
    power : float
        The power to raise x0 to (e.g., 1/3 for cube root, 0.25 for fourth root).

    Returns
    -------
    np.ndarray or float
        The result of the dilution operation.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**power)

def _concentrator(x0: ArrayLike) -> ArrayLike:
    """
    Applies a Concentrator operation (squaring) to fuzzy set memberships.
    Typically used to "sharpen" or narrow the meaning of a fuzzy set.

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the concentration (squaring) operation.
    """
    return x0**2

def _concentrator_power(x0: ArrayLike, power: int) -> ArrayLike:
    """
    Applies a generalized Concentrator operation (x0^power) to fuzzy set memberships.

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.
    power : int
        The integer power to raise x0 to (e.g., 3 for cubing, 4 for power of 4).

    Returns
    -------
    np.ndarray or float
        The result of the concentration operation.
    """
    return x0**power

def _fuzzy_and(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Calculates the fuzzy AND using the product (a * b) t-norm.

    Parameters
    ----------
    a : np.ndarray or float
        The first fuzzy set membership value or array.
    b : np.ndarray or float
        The second fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the fuzzy AND operation.
    """
    return a * b

def _fuzzy_or(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Calculates the fuzzy OR using the probabilistic sum (a + b - a*b) t-conorm.

    Parameters
    ----------
    a : np.ndarray or float
        The first fuzzy set membership value or array.
    b : np.ndarray or float
        The second fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the fuzzy OR operation.
    """
    return a + b - a * b

def _complement(x0: ArrayLike) -> ArrayLike:
    """
    Calculates the fuzzy complement (1 - x0).

    Parameters
    ----------
    x0 : np.ndarray or float
        The fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the fuzzy complement operation.
    """
    return 1 - x0

def _intensifier(x0: ArrayLike) -> ArrayLike:
    """
    Applies an Intensifier linguistic hedge (from "Expanding the definitions
    of linguistic hedges"). This operation increases membership values above 0.5
    and decreases those below 0.5, making the set "more true".

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the intensifier operation.
    """
    n = 2 # A common parameter for intensifier
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            x0 < 0,
            0,
            np.where(x0 < 0.5,
                     0.5**(1-n) * (x0**n),
                     1 - 0.5**(1-n) * (1 - x0)**n
            )
        )

def _diffuser(x0: ArrayLike) -> ArrayLike:
    """
    Applies a Diffuser linguistic hedge (from "Expanding the definitions
    of linguistic hedges"). This operation decreases membership values above 0.5
    and increases those below 0.5, making the set "less true" or "fuzzier".

    Parameters
    ----------
    x0 : np.ndarray or float
        The input fuzzy set membership value or array.

    Returns
    -------
    np.ndarray or float
        The result of the diffuser operation.
    """
    n = 2 # A common parameter for diffuser
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            x0 < 0,
            0,
            np.where(x0 < 0.5,
                     0.5**(1 - 1/n) * x0**(1/n),
                     1 - 0.5**(1 - 1/n) * (1 - x0)**(1/n)
            )
        )

def _if_gte(x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
    """
    Returns x1 if x1 >= x2, otherwise returns x2.

    Parameters
    ----------
    x1 : np.ndarray or float
        The first operand.
    x2 : np.ndarray or float
        The second operand (threshold).

    Returns
    -------
    np.ndarray or float
        The result based on the condition.
    """
    return np.where(x1 >= x2, x1, x2)

def _if_gte_else(x1: ArrayLike, x2: ArrayLike, x3: ArrayLike, x4: ArrayLike) -> ArrayLike:
    """
    Returns x3 if x1 >= x2, otherwise returns x4.

    Parameters
    ----------
    x1 : np.ndarray or float
        The comparison value.
    x2 : np.ndarray or float
        The threshold value.
    x3 : np.ndarray or float
        The value to return if the condition is true.
    x4 : np.ndarray or float
        The value to return if the condition is false.

    Returns
    -------
    np.ndarray or float
        The result based on the condition.
    """
    return np.where(x1 >= x2, x3, x4)

def _if_lt(x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
    """
    Returns x1 if x1 < x2, otherwise returns x2.

    Parameters
    ----------
    x1 : np.ndarray or float
        The first operand.
    x2 : np.ndarray or float
        The second operand (threshold).

    Returns
    -------
    np.ndarray or float
        The result based on the condition.
    """
    return np.where(x1 < x2, x1, x2)

def _if_lt_else(x1: ArrayLike, x2: ArrayLike, x3: ArrayLike, x4: ArrayLike) -> ArrayLike:
    """
    Returns x3 if x1 < x2, otherwise returns x4.

    Parameters
    ----------
    x1 : np.ndarray or float
        The comparison value.
    x2 : np.ndarray or float
        The threshold value.
    x3 : np.ndarray or float
        The value to return if the condition is true.
    x4 : np.ndarray or float
        The value to return if the condition is false.

    Returns
    -------
    np.ndarray or float
        The result based on the condition.
    """
    return np.where(x1 < x2, x3, x4)

def _lukasiewicz_t_norm(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """
    Calculates the Łukasiewicz t-norm: max(0, x0 + x1 - 1).

    Parameters
    ----------
    x0 : np.ndarray or float
        First value, typically in the range [0, 1].
    x1 : np.ndarray or float
        Second value, typically in the range [0, 1].

    Returns
    -------
    np.ndarray or float
        The Łukasiewicz t-norm of x0 and x1.
    """
    return np.maximum(0, x0 + x1 - 1)

def _lukasiewicz_t_conorm(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """
    Calculates the Łukasiewicz t-conorm: min(1, x0 + x1).

    Parameters
    ----------
    x0 : np.ndarray or float
        First value, typically in the range [0, 1].
    x1 : np.ndarray or float
        Second value, typically in the range [0, 1].

    Returns
    -------
    np.ndarray or float
        The Łukasiewicz t-conorm of x0 and x1.
    """
    return np.minimum(1, x0 + x1)

def _hamacher_t_norm(x0: ArrayLike, x1: ArrayLike, lambda_param: float) -> ArrayLike:
    """
    Computes the Hamacher T-norm.

    Parameters
    ----------
    x0 : np.ndarray or float
        First value, should be in the range [0, 1].
    x1 : np.ndarray or float
        Second value, should be in the range [0, 1].
    lambda_param : float
        Parameter lambda, should be >= 0.

    Returns
    -------
    np.ndarray or float
        The Hamacher T-norm of x0 and x1.

    Raises
    ------
    ValueError
        If inputs x0 or x1 are not in the range [0, 1] or if lambda_param is negative.
        Note: `np.any()` is used here for array inputs, but for element-wise fuzzy
        operations, it's often assumed inputs are already valid from upstream.
        Consider removing explicit checks if performance is critical and inputs
        are guaranteed by the framework.
    """
    # These checks are more robust if the inputs are single values or if
    # it's critical to halt execution for *any* out-of-range value in an array.
    # For general fuzzy set operations, often the domain [0,1] is assumed
    # by design, so explicit checks might be omitted for performance.

    if lambda_param < 0:
        raise ValueError("Parameter lambda_param must be non-negative.")

    numerator = x0 * x1
    denominator = lambda_param + (1 - lambda_param) * (x0 + x1 - x0 * x1)

    # Handle potential division by zero for denominator == 0
    # This specifically addresses the case where lambda_param is 0 and x0+x1-x0*x1 is also 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator == 0, 0, numerator / denominator)
        # Ensure result stays within [0, 1] range if any floating point inaccuracies occur
        return np.clip(result, 0, 1)

def _product_t_norm(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """
    Computes the product t-norm (x0 * x1). This is the standard fuzzy AND.

    Parameters
    ----------
    x0 : np.ndarray or float
        The first operand.
    x1 : np.ndarray or float
        The second operand.

    Returns
    -------
    np.ndarray or float
        The product of x0 and x1.
    """
    return x0 * x1

# --- gplearn Function Definitions ---
# These operators are wrapped for compatibility with gplearn's genetic programming framework.

MAXIMUM = functions.make_function(function=_maximum,
                                  name='MAXIMUM/or',
                                  arity=2)

MINIMUM = functions.make_function(function=_minimum,
                                  name='MINIMUM/and',
                                  arity=2)

COMPLEMENT = functions.make_function(function=_complement,
                                     name='COMPLEMENT/not',
                                     arity=1)

DILUTER = functions.make_function(function=_diluter,
                                  name='DILUTER',
                                  arity=1)

DILUTER3 = functions.make_function(function=lambda x0: _diluter_power(x0, 1/3),
                                   name='DILUTER3',
                                   arity=1)

DILUTER4 = functions.make_function(function=lambda x0: _diluter_power(x0, 0.25),
                                   name='DILUTER4',
                                   arity=1)

CONCENTRATOR = functions.make_function(function=_concentrator,
                                       name='CONCENTRATOR',
                                       arity=1)

CONCENTRATOR3 = functions.make_function(function=lambda x0: _concentrator_power(x0, 3),
                                        name='CONCENTRATOR3',
                                        arity=1)

CONCENTRATOR4 = functions.make_function(function=lambda x0: _concentrator_power(x0, 4),
                                        name='CONCENTRATOR4',
                                        arity=1)

FUZZY_AND = functions.make_function(function=_fuzzy_and,
                                    name='FUZZY_AND',
                                    arity=2)

FUZZY_OR = functions.make_function(function=_fuzzy_or,
                                   name='FUZZY_OR',
                                   arity=2)

INTENSIFIER = functions.make_function(function=_intensifier,
                                      name='INTENSIFIER',
                                      arity=1)

DIFFUSER = functions.make_function(function=_diffuser,
                                   name='DIFFUSER',
                                   arity=1)

IFGTE = functions.make_function(function=_if_gte,
                                name='IFGTE',
                                arity=2)

IFGTE2 = functions.make_function(function=_if_gte_else,
                                 name='IFGTE2',
                                 arity=4)

IFLT = functions.make_function(function=_if_lt,
                               name='IFLT',
                               arity=2)

IFLT2 = functions.make_function(function=_if_lt_else,
                                name='IFLT2',
                                arity=4)

LUKASIEWICZ_AND = functions.make_function(function=_lukasiewicz_t_norm,
                                      name='LUKASIEWICZ/AND',
                                      arity=2)

LUKASIEWICZ_OR = functions.make_function(function=_lukasiewicz_t_conorm,
                                      name='LUKASIEWICZ/OR',
                                      arity=2)

HAMACHER025 = functions.make_function(function=lambda x0, x1: _hamacher_t_norm(x0, x1, 0.25),
                                      name='HAMACHER025',
                                      arity=2)

HAMACHER050 = functions.make_function(function=lambda x0, x1: _hamacher_t_norm(x0, x1, 0.50),
                                      name='HAMACHER050',
                                      arity=2)

PRODUCT = functions.make_function(function=_product_t_norm,
                                  name='PRODUCT',
                                  arity=2)

# --- Dynamic generation of fixed-parameter WA and OWA operators ---
# This significantly reduces code duplication.

def _generate_wa_operator(param: float):
    """Generates a gplearn-compatible WA operator with a fixed weight."""
    def wa_func(a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return _weighted_average(a, b, param)
    wa_func.__name__ = f'_wa_op_param_{param}'.replace('.', '_')
    return wa_func

def _generate_owa_operator(param: float):
    """Generates a gplearn-compatible OWA operator with a fixed weight."""
    def owa_func(a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return _ordered_weighted_average(a, b, param)
    owa_func.__name__ = f'_wa_op_param_{param}'.replace('.', '_')
    return owa_func

# Create WA_P1 to WA_P9
for i in range(1, 10):
    param_val = round(i * 0.1, 1) # Calculate parameter value (0.1, 0.2, ..., 0.9)
    # Define the name as a string and use globals() to assign the dynamically created function
    globals()[f'WA_P{i}'] = functions.make_function(function=_generate_wa_operator(param_val),
                                                  name=f'WA_P{i}',
                                                  arity=2)

# Create OWA_P1 to OWA_P9
for i in range(1, 10):
    param_val = round(i * 0.1, 1) # Calculate parameter value (0.1, 0.2, ..., 0.9)
    globals()[f'OWA_P{i}'] = functions.make_function(function=_generate_owa_operator(param_val),
                                                   name=f'OWA_P{i}',
                                                   arity=2)
