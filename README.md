<table><tr><td><img style="float:left;padding-right:0px;vertical-align:top;border:none" src="https://raw.githubusercontent.com/pxom/zuffy/master/assets/zuffy_logo_small_nb_gr.png" alt="Zuffy Logo" width="80"/></td><td><h2>Zuffy - Fuzzy Pattern Trees with Genetic Programming</h2></td></tr></table>



## An open source scikit-learn compatible library for introducing FPTs as an Explainability Tool
------------------------------------------------------------------------------------------------
<!-- 
![tests](https://github.com/scikit-learn-contrib/project-template/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/scikit-learn-contrib/project-template/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/scikit-learn-contrib/project-template)
![doc](https://github.com/scikit-learn-contrib/project-template/actions/workflows/deploy-gh-pages.yml/badge.svg)
```diff
- NOTE THAT THIS PROJECT IS UNDER DEVELOPMENT AND LIKELY TO CHANGE SIGNIFICANTLY UNTIL THE FIRST RELEASE. USE AT YOUR OWN RISK.
```
-->
Zuffy is a python package for exploring Fuzzy Pattern Trees and it is compatible with [scikit-learn](https://scikit-learn.org).

It aims to provide a simple set of tools for the exploration of FPTs that are induced using 
genetic programming (GP) techniques. The GP functionality is provided by the gplearn library
and the author is grateful to its creator, Trevor Stephens, for his work on this.

Refer to the documentation for further information.

## Setup

Zuffy has been developed and tested with Python 3.11 and these library versions:

  Library    | Version  |
| ---------- | :------: |
| sklearn    | 1.5.2*   |
| numpy      | 1.26.4   |
| pandas     | 2.2.1    |
| matplotlib | 3.9.2    |
| gplearn    | 0.4.2    |

Note that Scikit-learn version 1.6+ modified the API around its "tags" and, until the authors update all their estimators, Zuffy will not run with version 1.6+.  The [gplearn library](https://github.com/trevorstephens/gplearn), upon which Zuffy has a dependency, has not been updated since 2023.

To display the FPT you will need to install [graphviz](https://graphviz.org/download/).


## Zuffy Installation
Clone the repository:
> git clone https://github.com/pxom/zuffy.git

Install the required dependencies:
> pip install -r requirements.txt

## Resources

- `Documentation <https://zuffy.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/zuffy-dev/zuffy/>`_
- `Installation <https://github.com/zuffy-dev/zuffy#installation>`_

## Examples

A collection of examples illustrating Zuffy features are available [here](<https://github.com/pxom/zuffy/tree/master/examples>).

## How to cite Zuffy
Authors of scientific papers including results generated using Zuffy are asked to cite the following paper:

```xml
@article{ZUFFY_1, 
    author    = "PXOM",
    title     = { {Zuffy}: Induction of Fuzzy Pattern Trees through Genetic Programming },
    pages    = { 0--0 },
    volume    = { 1 },
    month     = { Jul },
    year      = { 2025 }
}
```
