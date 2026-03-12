# Ill-Conditioning in Sparse Identification of Biological Dynamics

Code and data for the numerical experiments accompanying the paper on the effects of numerical ill-conditioning and multicollinearity in sparse, library-based discovery of governing equations from biological time-series data.


## Main contributions reproduced by this repository

The experiments in this repository are designed to support the following findings:

- Even small subsets of candidate functions can exhibit severe multicollinearity.
- Large condition numbers arise naturally in common biological benchmark systems.
- Sparse regularization alone does not eliminate sensitivity to ill-conditioning.
- Orthogonal polynomial libraries do not universally fix conditioning problems.
- When sampling is aligned with the weighting assumptions of the orthogonal basis, conditioning and recovery can improve substantially.


## Code descriptions
- Base_test.py - Define structures and simulations of baseline models.
- Basis.py - Define monomial and orthogonal bases including Chebyshev, Legendre and Laguerre basis used as the candidate function library in PySINDy.
- Collinearity.ipynb -
- Comparison.py - Identify terms in original and recovered equations, compare wrong and missing terms, and compute the condition number of the matrix constructed with wrong and missing terms.
- Definitions.py -
- Distance_test.ipynb -
- Functions.py -
- Jensen_Shannon.py -
- Models_exp.ipynb - Basic experiments for Beer model.
- Multi.py - 


## Related packages
- PySINDy
- DAE-FINDER

## Repository structure

This repository currently uses a lightweight script-and-notebook layout:

```text
.
├── README.md
├── .gitignore
├── Basis.py
├── Comparison.py
├── PolyConvert.py
├── tester.py
├── Untitled.ipynb
├── test.ipynb
├── testing.ipynb
├── Dependence/
├── Figure1/
├── Figure2/
├── Figure3/
└── Sethna Dataset1/
