# DS-NMF-with-SSVI

A doubly sparse extension of Non-negative Matrix Factorisation (NMF), described in the report: [Variational Inference for Bayesian Nonnegative Matrix Factorisation](https://vrs.amsi.org.au/student-profile/gyu-hwan-park/).

### Note:
This work extends the work by [Liang et al. (2014)](https://www.semanticscholar.org/paper/Beta-Process-Non-negative-Matrix-Factorization-with-Liang-Hoffman/34e4e531b19097505f73de8eaafdf3fd4e5fe797) where the model is sparse in only one of the factored matrices.

## What's included:

### code/
Contains the code for the Structured Stochastic Variational Inference algorithm for the Doubly Sparse NMF model.
* ds_ssvi: SSVI algorithm for DS-NMF
* s_ssvi: original SSVI algorithm for NMF by Liang et al. (2014), modified a little bit
* ds_ssvi_gibbs: Gibbs sampling version of the SSVI algorithm for DS-NMF

### notes/
Contains the report (with derivations in the Appendix) and short presentation slides about DS-NMF with SSVI.
