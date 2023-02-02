# XBART

## About

This package implements the local Gaussian process extrapolation for XBART. The manuscript is available [here](https://arxiv.org/abs/2204.10963). This approach builds on the methodology behind Accelerated Bayesian Additive Regression Trees (XBART) outlined in [He et al.](http://jingyuhe.com/files/xbart.pdf) (2019) and [He et al.](http://jingyuhe.com/files/scalabletrees.pdf) (2021).

This package is based on the source code of the [XBART](https://github.com/JingyuHe/XBART) package and was originally developed as a branch of that repository.

## Install Instruction

In terminal, type
```
cd XBART/tests/
bash test_py.sh
```

#### Trouble shooting
##### Mac

You might need to install the Xcode command line tools for compilers. (Not necessary to install the entire large Xcode software.)

Open a terminal, run 
```
xcode-select --install
```