ænet-PyTorch
============

## **ænet**

<span id="sec:about"></span>

The Atomic Energy NETwork (**ænet**) package (http://ann.atomistic.net) is a collection of tools for the construction and application of atomic interaction potentials based on artificial neural networks (ANN). ANN potentials generated with **ænet** can then be used in larger scale atomistic simulations.


## **ænet-PyTorch**

**ænet-PyTorch** is an extension of that code to allow GPU-support for the training process of **ænet**, substituting the `train.x` training step. It is enterily written in PyTorch and includes new features that the previous code did not: the ability to fit reference forces in addition to energies with GPU support. **ænet-PyTorch** is fully compatible with all the **ænet** tools: interfaces with LAMMPS and TINKER, and ASE.


# Installation

<span id="sec:installation"></span>

## Installation of **ænet**

The modified version of ænet can be installed the same way as ænet. See its documentation or any of the tutorials available in the ænet for a comprehensive guide on how to install it. In short, these are the steps to follow:

1.  Compile the L-BFGS-B library
      - Enter the directory “aenet_modified/lib”
        
        `$ cd aenet_modified/lib`
    - Adjust the compiler settings in the “Makefile” and compile the library with
        
        `$ make`
    
    The library file `liblbfgsb.a`, required for compiling **ænet**,  will be created.

2.  Compile the **ænet** package
    
      - Enter the directory “aenet_modified/src”
        
        `$ cd aenet_modified/src`
    
      - Compile the ænet source code with an approproiate `Makefile.XXX`
        
        `$ make -f makefiles/Makefile.XXX`
    
    The following executables will be generated in “./bin”:
    
      - `generate.x`: generate training sets from atomic structure files
      - `train.x`: train new neural network potentials
      - `predict.x`: use existing ANN potentials for energy/force prediction



## Installation of **ænet-PyTorch**

**ænet-PyTorch** is enterily written in Python, using the PyTorch framework. In the following the required packages will be listed. It is highly recommended to install all the packages in an isolated Python environment, using tools such as virtual-env (https://virtualenv.pypa.io/en/latest/) or a conda environment (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) .

  - `Python`: 3.7 or higher
  - `Numpy`: 1.19 or higher
  - `PyTorch`: 1.10 or higher
  - `CUDA`: 10.2 or higher (optional for GPU support) 

We will assume that `CUDA` and Python 3.7 are already installed.

1.  Install PyTorch 1.10

      - Installation using pip with CUDA support

        `$ pip install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`

        or for only CPU usage
    
        `$ pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html`


2.  Install Numpy

      - Numpy should be automatically installed when installing PyTorch.


## Installation of the tools for **ænet-PyTorch**

Compile the tools needed to make **ænet-PyTorch** compatible with **ænet**

  - Enter the directory "pytorch-aenet/tools"

    `cd tools`

  - Compile the tools

    `make`

The following exacutables will be generated in the same "tools/" directory:

  - `trainbin2ASCII.x`: convert the output from generate.x (`XXX.train`) to a format readable by **ænet-PyTorch** (`XXX.train.ascii`).
  - `nnASCII2bin.x`: convert the **ænet-PyTorch** output files (`XXX.nn.ascii`) to the usual binary files (`XXX.nn`).




# Usage

<span id="sec:usage"></span>

An explanation of the input parameters can be found in the documentation in the `doc/` directory. An example of the usage of the code can be found in the `example/` directory.