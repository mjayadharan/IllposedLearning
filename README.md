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
- Exp_LV_remark3.py & Exp_CRN_remark3.py - Assess robustness to mismatch between the empirical sampling distribution and the target distribution by varying the number of initial conditions and trajectory length.
- remark3_experiment.py - Supporting files for Exp_LV_remark3.py and Exp_CRN_remark3.py.
- Exp_IC_condition_robustness.py - Assess the robustness of the results across different IC settings and their dependence on the sampling scheme.
- Exp_library_scaling_comparison.py - Compare condition numbers between libraries with columns normalized to unit L2 norm and libraries at their original scale. (sampled data)
- Exp_library_scaling_comparison_original_data.py - Compute the condition numbers of different polynomial libraries for the two baseline models. (original data)
- Exp_original_raw_sensitivity.py - Perform a sensitivity analysis of SINDy model recovery with respect to the sparsity threshold and ridge penalty.
- Exp_table1_lv_crn_l2.py - Compute the condition numbers reported in Tables SM1–SM4 of the supplementary material.
- Exp_Plot3_condition.py - Compute the condition number of each benchmark model using scaled library.

Figure 1 is generated primarily using Plots-1_modify.ipynb and Exp_library_scaling_comparison_original_data.py. The former generates panels (a)–(b) and (e)–(f), while the latter generates panels (c)–(d).

Figure 2 is generated using Exp_Plot3_condtion.py.

Figure 3 is generated primarily using Plots_2.ipynb and Exp_library_scaling_comparison.py. The former generates panels (a)-(d) and (g)-(h), while the latter generates panels (e)-(f).

Figure 4 is generated using Exp_LV_remark3.py & Exp_CRN_remark3.py. The former generates panels (a)-(d), while the latter generates panels (e)-(h).


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
