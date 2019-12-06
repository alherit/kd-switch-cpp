# kd-switch (C++ with Python interface)

This repository contains an optimized iterative C++ implementation of the **kd-switch** online predictor proposed in our paper:

    Low-Complexity Nonparametric Bayesian Online Prediction with Universal Guarantees
    Alix Lhéritier & Frédéric Cazals
    NeurIPS 2019

[ArXiv version](https://arxiv.org/abs/1901.07662)

This version runs about 70X faster than our pure Python recursive implementation https://github.com/alherit/kd-switch.

This C++ implementation is also accessible from Python using the provided bindings.

## Dependencies

* GCC with C++11 support
* Cmake >= 2.8.12

For Python binding: 
* Python >= 3.4

For example.py : 
* Numpy and scikit-learn


## Compilation  

```
mkdir build
cd build
cmake ..
make
```

Add build directory to PYTHONPATH if you want to use the KDSForest class from Python.

## Examples

See example.cpp and example.py.

## Note about the "frozen" option

By default, *predict* has a standard behavior, i.e.: it doesn't affect the model, only *update* does.
If *frozen=False* is specified, then *predict* does some of the work for the next update to come that is assumed to use the same point: this avoids doing twice the prediction part.
If *frozen=False* is used, it must be specified both in the predict call and in the following update call.

## License
[MIT license](https://github.com/alherit/kd-switch-cpp/blob/master/LICENSE).

If you have questions or comments about anything regarding this work, please see the paper for contact information.
