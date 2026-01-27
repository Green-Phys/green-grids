[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-grids?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![grids](https://github.com/Green-Phys/green-grids/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Green-Phys/green-grids/graph/badge.svg?token=Q4L185DC33)](https://codecov.io/gh/Green-Phys/green-grids)

# Sparse grid utilities for IR and Chebyshev representations

## Essential contents for a temperature-independent sparse grid

- `stats`: Bose or Fermi statistics.
- `xgrid`: Real space grid points in $x \in (-1, 1)$.
- `ngrid`: Frequency space grid points in Matsubara indices $n$, such that
  $\omega_n = (2n+\zeta)\pi/\beta$.
- `wgrid`: Frequency space grid points assuming $\beta=1$.
- `uxl`: Transformation matrix from the basis representation `l` to real space
  grid points `x`.
- `ulx`: Inverse of `uxl`.
- `u1l_pos`: Basis representation for $x=+1$.
- `u1l_neg`: Basis representation for $x=-1$.
- `uwl`: Transformation matrix from the basis representation `l` to frequency
  space grid points `w`.
- `ulw`: Inverse of `uwl`.
- `metadata`: Group for representation-depend information, for example:
    - For Chebyshev: `ncoeff` for maximum number of coefficients.
    - For IR: `lambda` for dimensionless cutoff $\Lambda$, `ncoeff` for maximum
      number of coefficients.

## Directory structure

- `python/sparse_grid/`: python code for generating and parsing sparse grid data.
    - `python/sparse_grid/ir/`: IR-specific code.
    - `python/sparse_grid/chebyshev/`: Chebyshev-specific code.
    - `python/sparse_grid/repn.py`: Common module interface for representations.
    - `python/generate.py`: Script for generating HDF5 archives.
- `c++/`: C++ interface for loading and representing sparse grid data.
    - `green/grids/`: Public headers.
- `data/`: Pre-generated data files.
- `examples`: Usage examples.

## Dependencies

- Python:
    - [`sparse-ir`](https://github.com/SpM-lab/sparse-ir)
    - `numpy`, `scipy`, `h5py`, and `mpmath`.
- C++:
    - Green/h5pp: for compatibility with h5py
    - Green/ndarrays: for compatibility with numpy.ndarray
    - Green/params: for comandline parameters
    - CMake: Version 3.27 or later

## Installation

C++ installation uses CMake in a straightforward manner:
```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make -j 4
make install
make test  # Test the code
```

The Python package for `green-grids` can be installed using PyPI:
```bash
pip install green-grids
```
or simply by building from source as:
```bash
git clone https://github.com/Green-Phys/green-grids
cd green-grids
pip install .
```

# Acknowledgements

This work is supported by National Science Foundation under the award CSSI-2310582
