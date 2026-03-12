# Ill-Conditioning in Sparse Identification of Biological Dynamics

Code and data for the numerical experiments accompanying the paper on the effects of numerical ill-conditioning and multicollinearity in sparse, library-based discovery of governing equations from biological time-series data.


## Main contributions reproduced by this repository

The experiments in this repository are designed to support the following findings:

- Even small subsets of candidate functions can exhibit severe multicollinearity.
- Large condition numbers arise naturally in common biological benchmark systems.
- Sparse regularization alone does not eliminate sensitivity to ill-conditioning.
- Orthogonal polynomial libraries do not universally fix conditioning problems.
- When sampling is aligned with the weighting assumptions of the orthogonal basis, conditioning and recovery can improve substantially.


# Related packages
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
