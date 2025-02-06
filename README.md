<table><tr><td><img style="float:left;padding-right:0px;vertical-align:top;border:none" src="assets/zuffy_logo_small_nb_gr.png" alt="Zuffy Logo" width="80"/></td><td><h2>Zuffy - Fuzzy Pattern Trees with Genetic Programming</h2></td></tr></table>


## A Scikit-learn compatible Open Source library for introducing FPTs as an Explainability Tool
------------------------------------------------------------------------------------------
<!-- 
![tests](https://github.com/scikit-learn-contrib/project-template/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/scikit-learn-contrib/project-template/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/scikit-learn-contrib/project-template)
![doc](https://github.com/scikit-learn-contrib/project-template/actions/workflows/deploy-gh-pages.yml/badge.svg)
-->

### (NOTE THAT THIS PROJECT IS UNDER DEVELOPMENT AND LIKELY TO CHANGE SIGNIFICANTLY UNTIL THE FIRST RELEASE. USE AT YOUR OWN RISK.)
Zuffy is an open source python library for explainable machine learning models.  It is compatible with [scikit-learn](https://scikit-learn.org).

It aims to provide a simple set of tools for the exploration of FPTs that are inferred using 
genetic programming techniques.

Refer to the documentation for further information.

## Setup

It may work with other versions but Zuffy has been tested with Python 3.11.9 and these library versions:

  Library    | Version  |
| ---------- | :------: |
| sklearn    | 1.5.2*   |
| numpy      | 1.26.4   |
| pandas     | 2.2.1    |
| matplotlib | 3.9.2    |
| gplearn    | 0.4.2    |

Note that Scikit-learn version 1.6+ modified the API around its "tags" and, until the authors update all their estimators, Zuffy will not run with version 1.6+.

To display the FPT you will need to install graphviz:

##### Unix
```bash 
sudo apt install graphviz
```

> $ sudo apt install graphviz

##### Windows
???


## Installation
Clone the repository:
> git clone https://github.com/pxom/zuffy.git
Install the required dependencies:
> pip install -r requirements.txt

## Resources

- `Documentation <https://zuffy.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/zuffy-dev/zuffy/>`_
- `Installation <https://github.com/zuffy-dev/zuffy#installation>`_

## Examples

To see more elaborate examples, look [here](<https://github.com/pxom/zuffy/tree/master/notebooks/README.md>).


```python

import pandas as pd
from sklearn.datasets import load_iris
from zuffy import ZuffyClassifier, functions, visuals
from zuffy.wrapper import ZuffyFitIterator

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target
targetNames = iris.target_names
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X)

zuffy = ZuffyClassifier(generations=15, verbose=1)
res = ZuffyFitIterator(zuffy, fuzzy_X, y, n_iter=3, split_at=0.25)

visuals.plot_evolution(
    res.getBestEstimator(),
    targetNames,
    res.getPerformance(),
    outputFilename='sample1_analysis')

visuals.graphviz_tree(
    res.getBestEstimator(),
    targetNames,
    featureNames=fuzzy_features_names,
    treeName=f"Iris Dataset (best accuracy: {res.getBestScore():.3f})",
    outputFilename='sample1_fpt')
```

### * TBD *
In an `sklearn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_:

```python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)
```


## How to cite Zuffy
Authors of scientific papers including results generated using Zuffy are asked to cite the following paper.

```xml
@article{ZUFFY_1, 
    author    = "POM",
    title     = { {Zuffy}: Open Source inference of FPT using GP },
    pages    = { 0--0 },
    volume    = { 1 },
    month     = { Apr },
    year      = { 2025 },
    journal   = { Journal of Unknown }
}
```
