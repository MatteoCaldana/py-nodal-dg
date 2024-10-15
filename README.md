# Nodal Discontinuous Galerkin Methods in Python

## Overview

This repository provides a Python implementation of the Nodal Discontinuous Galerkin (DG) methods for solving partial differential equations (PDEs) as described in the book *Nodal Discontinuous Galerkin Methods* by Jan Hesthaven and Tim Warburton. The goal of this project is to recreate the functionality of the original MATLAB routines in a Python environment, making it easy to integrate a automatic differentiation framework.

Indeed, this code is the core of the algorithm developed in:
*Caldana, M., Antonietti, P.F. and Dede' L., 2024. Discovering artificial viscosity models for discontinuous galerkin approximation of conservation laws using physics-informed machine learning. Journal of Computational Physics, p.113476.*

## Examples

The `examples` directory contains several example scripts demonstrating the usage of the implemented methods. To run the examples you must install the library with `pip install -e .`.

## References

- Hesthaven, J.S., & Warburton, T. (2008). Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications. Springer.
- [Original MATLAB code repository](https://github.com/tcew/nodal-dg)
- [MATLAB based DG-solver for conservation laws](https://github.com/nickdisca/DGANN_AV)
