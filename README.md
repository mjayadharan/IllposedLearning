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
- Comparison.py - Identify terms in original and recovered equations, compare wrong and missing terms, and compute the condition number of the matrix constructed with wrong and missing terms.
- Models_exp.ipynb - Basic experiments for Beer model.
- Multicollinearity.py - Analysis of ill-posed combinations.
- PolyConvert.py - Convert orthogonal polynomials to monomials and denormalize the recovered equations for the Legendre and Chebyshev bases.
- Recover_Model.py - Recover models from data using monomial and orthogonal polynomial bases.
- Sample_Analysis.py - Sample data according to specific distributions.

Basis.py is used to generate Fig.1(c) and Fig.1(d); Comparison.py is used to generate Fig.1(e) and Fig.1(f).

Sampling_Analysis.py is used to generate Fig.2(a)-(d); Basis.py is used for Fig.2(e)-(f); Sample_Analysis.py and Recover_Model.py are used for Fig.2(g)-(h).

The results for benchmark models are shown in Fig.3. The condition numbers of the full library, as well as those of the incorrect and missing terms, are computed following the same procedure illustrated in Figs.1 and 2.

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
