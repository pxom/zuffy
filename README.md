![Zuffy Logo](https://github.com/pxom/fptgp/blob/master/assets/zuffy_logo_small.png)

<!-- [logo](https://github.com/pxom/fptgp/blob/master/assets/zuffy_logo.png) "Why are you hovering? Go for it!" -->

Zuffy - Fuzzy Pattern Trees with Genetic Programming
====================================================
Scikit-learn compatible Open Source library for introducing FPTs as an Explainability Tool
------------------------------------------------------------------------------------------

<!-- 
![tests](https://github.com/scikit-learn-contrib/project-template/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/scikit-learn-contrib/project-template/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/scikit-learn-contrib/project-template)
![doc](https://github.com/scikit-learn-contrib/project-template/actions/workflows/deploy-gh-pages.yml/badge.svg)
-->

Zuffy is an open source python library for explainable machine learning models that is compatible with scikit-learn [scikit-learn](https://scikit-learn.org).

It aims to provide a simple set of tools for the user to explore FPTs that are inferred using 
genetic programming techniques.

Refer to the documentation for further information.

## Setup
To set up:
* pip install pandas
* pip install scikit-learn # not 1.6.1 - developed on windows with 1.4.1.post1.

    pip uninstall scikit-learn
    administrator@lt-tishdevt:/mnt/code$ 
        pip install scikit-learn-1-4-1.post1.
        Defaulting to user installation because normal site-packages is not writeable
        ERROR: Invalid requirement: 'scikit-learn-1-4-1.post1.'

        Scikit-learn version 1.6 modified the API around its "tags", and that's the cause of this error.
        
        pip index versions scikit-learn
        pip uninstall scikit-learn


        # set up remote installer
        python3 -m pip install --upgrade debugpy

        sudo pip install matplotlib --user


pip install gplearn  # version 0.4.2
pip install matplotlib
pip install graphviz


=========
Resources
=========

- `Documentation <https://skorch.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/skorch-dev/skorch/>`_
- `Installation <https://github.com/skorch-dev/skorch#installation>`_

========
Examples
========

To see more elaborate examples, look `here
<https://github.com/skorch-dev/skorch/tree/master/notebooks/README.md>`__.

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from torch import nn
    from skorch import NeuralNetClassifier

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    class MyModule(nn.Module):
        def __init__(self, num_units=10, nonlin=nn.ReLU()):
            super().__init__()

            self.dense0 = nn.Linear(20, num_units)
            self.nonlin = nonlin
            self.dropout = nn.Dropout(0.5)
            self.dense1 = nn.Linear(num_units, num_units)
            self.output = nn.Linear(num_units, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            X = self.nonlin(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    net = NeuralNetClassifier(
        MyModule,
        max_epochs=10,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    net.fit(X, y)
    y_proba = net.predict_proba(X)

In an `sklearn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)


*Thank you for cleanly contributing to the scikit-learn ecosystem!*
