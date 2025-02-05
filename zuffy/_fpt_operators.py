"""
@author: Peter O'Mahony <fptgp@mahoonium.com>
License: BSD 3 clause
This module contains the default set of FPT operators.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gplearn import functions

def WA(a, b, x):
    """
    Weighted Average: This returns x*a + (1-x)*b

    NB: This should be highlighted in an Attention box.

    Warning: This should be in a Warning box.

    Parameters
    ----------
    a: np.array, default=None
        The first matrix to be compared.

    b: np.array, default=None
        The second matrix to be compared.

    x: float, default=None
        The weight to apply to matrix a and the complement of which is applied to b.

    Attributes
    ----------
    root: Node
        This a sample.

    _impurity_gain_calculation_func: function
        This a sample.

    References
    1. Ho, T.K., 1995, August. Random decision forests. In Proceedings
        of 3rd international conference on document analysis and
        recognition (Vol. 1, pp. 278-282). IEEE.
    2. Ho, T.K., 1998. The random subspace method for constructing
        decision forests. IEEE transactions on pattern analysis and
        machine intelligence, 20(8), pp.832-844.                
    """
    x = float(x)
    return x*a+(1-x)*b

def OWA(a, b, x):
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

def minimum(a, b):
    return np.minimum(a, b)

def maximum(a, b):
    return np.maximum(a, b)
    
def DILUTER(b):
    return b**0.5

def DILUTER3(b):
    return b**(1/3)

def DILUTER4(b):
    return b**0.25

def concentrator(b):
    return b**2

def concentrator3(b):
    return b**3

def concentrator4(b):
    return b**4

def fuzzy_AND(a, b):
    return a * b

def fuzzy_OR(a, b):
    return a + b - a*b

def complement(b):
    return 1 - b

def _MAXIMUM(x0, x1):
    return np.maximum(x0, x1)

MAXIMUM = functions.make_function(function=_MAXIMUM,
                        name='MAXIMUM/or',
                        arity=2)

def _MINIMUM(x0, x1):
    return np.minimum(x0, x1)

MINIMUM = functions.make_function(function=_MINIMUM,
                        name='MINIMUM/and',
                        arity=2)

def _COMPLEMENT(x0):
    return 1 - x0

COMPLEMENT = functions.make_function(function=_COMPLEMENT,
                        name='COMPLEMENT/not',
                        arity=1)

def _DILUTER(x0):
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**0.5)

DILUTER = functions.make_function(function=_DILUTER,
                        name='DILUTER',
                        arity=1)

def _DILUTER2(x0):
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**0.25)

DILUTER2 = functions.make_function(function=_DILUTER2,
                        name='DILUTER2',
                        arity=1)
'''
def _CONCENTRATOR3(x0):
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**0.125)

CONCENTRATOR3 = functions.make_function(function=_CONCENTRATOR3,
                        name='CONCENTRATOR3',
                        arity=1)

def _CONCENTRATOR4(x0):
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**(1/16))

CONCENTRATOR4 = functions.make_function(function=_CONCENTRATOR4,
                        name='CONCENTRATOR4',
                        arity=1)

def _CONCENTRATOR8(x0):
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x0 < 0, 0, x0**(1/32))

CONCENTRATOR8 = functions.make_function(function=_CONCENTRATOR8,
                        name='CONCENTRATOR8',
                        arity=1)
'''

def _CONCENTRATOR(x0):
    return x0**2

CONCENTRATOR = functions.make_function(function=_CONCENTRATOR,
                        name='CONCENTRATOR',
                        arity=1)

def _CONCENTRATOR2(x0):
    return x0**4

CONCENTRATOR2 = functions.make_function(function=_CONCENTRATOR2,
                        name='CONCENTRATOR2',
                        arity=1)

def _INTENSIFIER(x0): # from Expanding the definitions of linguistic hedges
    """Closure of division by for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        n = 2
        return np.where(
                x0 < 0,
                0,
                np.where(x0 < 0.5,
                         0.5**(1-n) * (x0**n),
                         1 - 0.5**(1-n) * (1 - x0)**n
                )
        )

INTENSIFIER = functions.make_function(function=_INTENSIFIER,
                        name='INTENSIFIER',
                        arity=1)


def _DIFFUSER(x0): # from Expanding the definitions of linguistic hedges
    n = 2
    with np.errstate(divide='ignore', invalid='ignore'):
        n = 2
        return np.where(
                x0 < 0,
                0,
                np.where(x0 < 0.5,
                     0.5**(1 - 1/n) * x0**(1/n),
                     1 - 0.5**(1-1/n) * (1 - x0)**(1/n)
                )
        )

DIFFUSER = functions.make_function(function=_DIFFUSER,
                        name='DIFFUSER',
                        arity=1)



def _SUB_AMT_025(x1):
    return x1 - 0.25

SUB_AMT_025 = functions.make_function(function=_SUB_AMT_025,
                        name='SUB_AMT_025',
                        arity=1)

def _IFGTE(x1, x2):
    return np.where(x1 >= x2, x1, x2)

IFGTE = functions.make_function(function=_IFGTE,
                        name='IFGTE',
                        arity=2)

def _IFGTE2(x1, x2, x3, x4):
    return np.where(x1 >= x2, x3, x4)

IFGTE2 = functions.make_function(function=_IFGTE2,
                        name='IFGTE2',
                        arity=4)

def _IFLT(x1, x2):
    return np.where(x1 < x2, x1, x2)

IFLT = functions.make_function(function=_IFLT,
                        name='IFLT',
                        arity=2)

def _IFLT2(x1, x2, x3, x4):
    return np.where(x1 < x2, x3, x4)

IFLT2 = functions.make_function(function=_IFLT2,
                        name='IFLT2',
                        arity=4)

def _WA_P1(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P1 = functions.make_function(function=_WA_P1,name='WA_P1',arity=2)

def _WA_P2(a, b):
    x = 0.2
    x = float(x)
    return x*a+(1-x)*b

WA_P2 = functions.make_function(function=_WA_P2,name='WA_P2',arity=2)

def _WA_P3(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P3 = functions.make_function(function=_WA_P3,name='WA_P3',arity=2)

def _WA_P4(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P4 = functions.make_function(function=_WA_P4,name='WA_P4',arity=2)

def _WA_P5(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P5 = functions.make_function(function=_WA_P5,name='WA_P5',arity=2)

def _WA_P6(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P6 = functions.make_function(function=_WA_P6,name='WA_P6',arity=2)

def _WA_P7(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P7 = functions.make_function(function=_WA_P7,name='WA_P7',arity=2)

def _WA_P8(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P8 = functions.make_function(function=_WA_P8,name='WA_P8',arity=2)

def _WA_P9(a, b):
    x = 0.1
    x = float(x)
    return x*a+(1-x)*b

WA_P9 = functions.make_function(function=_WA_P9,name='WA_P9',arity=2)


def _OWA_P1(a, b):
    x = 0.1
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P1 = functions.make_function(function=_OWA_P1,name='OWA_P1',arity=2)

def _OWA_P2(a, b):
    x = 0.2
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P2 = functions.make_function(function=_OWA_P2,name='OWA_P2',arity=2)

def _OWA_P3(a, b):
    x = 0.3
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P3 = functions.make_function(function=_OWA_P3,name='OWA_P3',arity=2)

def _OWA_P4(a, b):
    x = 0.4
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P4 = functions.make_function(function=_OWA_P4,
                        name='OWA_P4',
                        arity=2)

def _OWA_P5(a, b):
    x = 0.5
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P5 = functions.make_function(function=_OWA_P5,name='OWA_P5',arity=2)

def _OWA_P6(a, b):
    x = 0.6
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P6 = functions.make_function(function=_OWA_P6,name='OWA_P6',arity=2)

def _OWA_P7(a, b):
    x = 0.7
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P7 = functions.make_function(function=_OWA_P7,name='OWA_P7',arity=2)

def _OWA_P8(a, b):
    x = 0.8
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P8 = functions.make_function(function=_OWA_P8,name='OWA_P8',arity=2)

def _OWA_P9(a, b):
    x = 0.9
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

OWA_P9 = functions.make_function(function=_OWA_P9,name='OWA_P9',arity=2)

#def _POINT_1(x):
#    x = np.ones(x) * 0.1
#    x = 0.1
#    return x

#POINT_1 = functions.make_function(function=_POINT_1,
#                        name='POINT_1',
#                        arity=1)

# these WTA functions don't work because they don't return numpy arrays (which I fixed) and then don't return the same shape as input vectors so figure it out!
def _WTA(a, b):
    #return [a, b]
    return [np.array(a), np.array(b)] # pom added no.array

#WTA = functions.make_function(function=_WTA, name='WTA', arity=2)

def _WTA3(a, b, c):
    return [a, b, c]
    #return np.array([a, b, c]) # pom added no.array

#WTA3 = functions.make_function(function=_WTA3, name='WTA3', arity=3)
