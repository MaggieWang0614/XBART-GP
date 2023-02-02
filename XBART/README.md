# XBART

## About

This package implements the local Gaussian process extrapolation for XBART. The manuscript is available [here](https://arxiv.org/abs/2204.10963). This approach builds on the methodology behind Accelerated Bayesian Additive Regression Trees (XBART) outlined in [He et al.](http://jingyuhe.com/files/xbart.pdf) (2019) and [He et al.](http://jingyuhe.com/files/scalabletrees.pdf) (2021).

This package is based on the source code of the [XBART](https://github.com/JingyuHe/XBART) package and was originally developed as a branch of that repository.

## Install Instruction

In terminal, type
```
cd tests/
bash test_r.sh
```
An up-to-date version of the package can be installed in R console:
```R
library(devtools)
install_github("JingyuHe/XBART")
```

#### Trouble shooting

##### Windows

If you have any compiler error, please install the latest version of Rtools [Link](https://cran.r-project.org/bin/windows/Rtools/rtools42/rtools.html), it will install all necessary compilers and dependencies.

##### Mac

You might need to install the Xcode command line tools for compilers. (Not necessary to install the entire large Xcode software.)

Open a terminal, run 

```R
xcode-select --install
```

##### Linux

Linux is already shipped with all necessary compilers. Since you are using Linux, you must be an expert ;-)

##### GSL

If you can't in stall it on Mac becase of an error message says 'gsl/gsl_sf_bessel.h' not found. Try following steps.

1, Open a terminal, run ```brew install gsl```.

2, Check if gsl is installed in the following directory: /opt/homebrew/Cellar/gsl/2.7.1 (if not, it should be somewhere similar, try searching for gsl).

3, In terminal, type 
```
cd ~/.R
open Makevars
```
(If you don’t have the Makevars file, create one by ```touch Makevars```)

4, in the file, paste in:
```
LDFLAGS+=-L/opt/homebrew/Cellar/gsl/2.7.1/lib
CPPFLAGS+=-I/opt/homebrew/Cellar/gsl/2.7.1/include
```
or with equivalent directory where your gsl library is installled.
