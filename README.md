<table><tr><td><img style="float:left;padding-right:0px;vertical-align:top;border:none" src="assets/zuffy_logo_small_nb.png" alt="Zuffy Logo" width="80"/></td><td><h2>Zuffy - Fuzzy Pattern Trees with Genetic Programming</h2></td></tr></table>


## A Scikit-learn compatible Open Source library for introducing FPTs as an Explainability Tool
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

It may work with other versions but zuffy has been tested with these versions:

  Library    | Version  |
| ---------- | :------: |
| sklearn    | 1.5.2*   |
| numpy      | 1.26.4   |
| pandas     | 2.2.1    |
| matplotlib | 3.9.2    |
| gplearn    | 0.4.2    |

Note that Scikit-learn version 1.6+ modified the API around its "tags" and, until the authors update all their estimators, zuffy will not run with version 1.6+.

To display the FPT you will need to install graphviz:

##### Unix
```bash 
sudo apt install graphviz
```

> $ sudo apt install graphviz

##### Windows
???

## Resources

- `Documentation <https://zuffy.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/zuffy-dev/zuffy/>`_
- `Installation <https://github.com/zuffy-dev/zuffy#installation>`_

## Examples

To see more elaborate examples, look `here
<https://github.com/zuffy-dev/zuffy/tree/master/notebooks/README.md>`__.

```python

    import numpy as np
    from sklearn.datasets import make_classification
    from torch import nn

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
```

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
