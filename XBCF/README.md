# Accelerated Bayesian Causal Forests (XBCF)

## About

This package implements the local Gaussian process extrapolation for XBCF. The manuscript is available [here](https://arxiv.org/abs/2204.10963). This approach builds on the methodology behind Accelerated Bayesian Causal Forests (XBCF) outlined in [Krantsevich et al.](https://math.la.asu.edu/~prhahn/XBCF.pdf).

This package is based on the source code of the [XBCF](https://github.com/socket778/XBCF) package and was originally developed as a branch of that repository.


## Installation

'```
cd tests/
bash test_r.sh
```

An up-to-date version of the package can be installed in R console:
```R
library(devtools)
install_github("socket778/XBCF", branch = "xbcf-gp")
```

