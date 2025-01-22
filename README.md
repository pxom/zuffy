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

To see more elaborate examples, look [here](<https://github.com/zuffy-dev/zuffy/tree/master/notebooks/README.md>).


```python

import pandas as pd
from sklearn.datasets import load_iris
from fptgp.fptgp import FPTGPClassifier, functions, visuals
from fptgp.fptgp.wrapper import FPTGP_fit_iterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
targetNames = iris.target_names
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X)

fptgp = FPTGPClassifier(generations=15, verbose=1)
res = FPTGP_fit_iterator(fptgp, fuzzy_X, y, n_iter=3, split_at=0.25, random_state=77)

visuals.plot_evolution(
    res.getBestEstimator(),
    targetNames,
    res.getPerformance(),
    outputFilename='sample1_analysis')

visuals.graphviz_tree(
    res.getBestEstimator(),
    targetNames,
    featureNames=fuzzy_features_names,
    treeName="Iris Dataset (best accuracy: " + str(round(res.getBestScore(),3)) + ")",
    outputFilename='sample1')
```

In an `sklearn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_:

```python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)```


## How to cite Zuffy
Authors of scientific papers including results generated using Zuffy are asked to cite the following paper.

```xml
@article{ZUFFY_1, 
    author    = "Peter O'Mahony",
    title     = { {Zuffy}: Open Source inference of FPT using GP },
    pages    = { 0--0 },
    volume    = { 1 },
    month     = { Apr },
    year      = { 2025 },
    journal   = { Journal of Unknown }
}
```
