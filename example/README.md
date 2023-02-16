Training with ænet-PyTorch
==========================

This directory contains an example that shows the training of ANNs (fitted to both reference energy and forces) using ænet-PyTorch.

In this example we will use is a small subset of the reference data set used is the TiO<sub>2</sub> data set from [*Comput. Mater. Sci.* **114** (2016) 135-150](http://dx.doi.org/10.1016/j.commatsci.2015.11.047). The data used is compressed in `dir_xsf.tar.gz`.

The input files used in each step and the output files generated are contained in each subdirectory.

01-generate
-----------

Evaluation of the Chebyshev descriptor for all structures in the reference data set.  The resulting feature vectors are written to a *training set* file. Only 25% of the structures will be considered as reference for force training.

#### Input ####

- `generate.in`: Input file for `generate.x`
- `dir_xsf/`: Directory with the `xsf` files of the reference data set.
- `O.fingerprint.stp` and `Ti.fingerprint.stp`: Descriptor definitions for the atomic species O and Ti.

#### Execution ####

- Evaluate of the desciptors for all the structures

    `aenet_modified/bin/generate.x generate.in`

- Convert the output to the **ænet-PyTorch**-readable format 

    `tools/trainbin2ASCII.x TiO.train TiO.train.ascii`

#### Output ####

*Training set* files including information about the descriptors and their derivatives (not included because of the file size).

  - `TiO.train.ascii`: Descriptors and training set information
  - `TiO.train.forces`: Derivatives of the descriptors for the structures selected for training forces.


02-train
--------

Training example using the *training set* files generated in the previous step.

#### Input ####

- `train.in`: Input file for `ænet-PyTorch`
- `TiO.train.ascii` and `TiO.train.forces`: *Training set* files.

#### Execution ####

- Train the neural networks using **ænet-PyTorch**

    `python3 aenet_modified/src/aenet_pytorch.py`

- Convert the output to the **ænet** binary format 

    `tools/nnASCII2bin.x Ti.pytorch.nn.ascii Ti.pytorch.nn`
    `tools/nnASCII2bin.x  O.pytorch.nn.ascii  O.pytorch.nn`

#### Output ####

ANN potential files `O.pytorch.nn` and `Ti.pytorch.nn`.


03-predict
----------

Usage of the ANN potentials trained in the previous step for the prediction of the reference data set. This is the same as in **ænet**

#### Input ####

- `predict.in`: Input file for `predict.x`
- `O.pytorch.nn` and `Ti.pytorch.nn`: ANN potential files

#### Execution ####

- Predict energy/forces for all the structures in `predict.in`

    `aenet_modified/bin/predict.x predict.in`

#### Input ####

The output (energies and atomic forces) can be found in the `output` subdirectory.