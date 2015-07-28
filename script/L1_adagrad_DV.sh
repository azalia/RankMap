#!/bin/bash

# This code performs L_2 minimizarion with L1 penalty on factrized data matrix DV: x = argmin: ||Ax-y||_2 + lambda*|x|_1  
# Where A is approximated by D*V. D is a dictionary matrix and V is a coefficient matric. The factorization can be achieved using the RankMap methodology or other factorization schemes.
# We use Adaptive Subgradient Methods (Adagrad) for gradient tuning


#Notes for comiling and executing:

#For compiling modules GCC, OpenMPI (or alternatives) should be loaded, and the EIGEN library should be installed


#For compiling the code:
mpic++  -o outpuname.o ../src/L1_adagrad_DV.cpp
#For running the code:
mpiexec -n npes outpuname.o "V directory" "D directory" "y directory" Nf M N L Ns Maxiter Lambda Verbose 

#Expression of variables:

# Matrix V should be written in a row col value format. Each line corresponds to one element in V.
# Matrix D should be written in a general one line per row matrix format.
# Nf: number of files V is storred in, e.g., 1,2,.... If V is large, it can be storred in multiple files. The V file names should end  in _0, _1, _2, ..._Nf.
# M: number of rows in A (or D)
# N: number of columns in A (or V)
# L: number of columns in D
# Ns: number of columns in y
# Maxiter: maximum number of gradien descent iteraions
# Lambda : L1 penalty coefficient
# Verbose:  set to 1 to see reports, set to 0 otherwise. 


#  The output x files are written in  "D directory"_Xopt_lamda_npes_id files. where id = 0,1,2,...,npes-1  



