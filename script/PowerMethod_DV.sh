#!/bin/bash

# This code performs power method on factrized data matrix DV
# The original matrix A is approximated by D*V. D is a dictionary matrix and V is a coefficient matric. The factorization can be achieved using the RankMap methodology or other factorization schemes.
# We use power method iterative updates on the Gram matrix (DV)^TDV


#Notes for comiling and executing:

#For compiling modules GCC, OpenMPI (or alternatives) should be loaded, and the EIGEN library should be installed


#For compiling the code:
mpic++  -o outpuname.o ../src/powerMethod_DV.cpp
#For running the code:

srun -n npes outpuname.o "V directory" "D directory" Nf M N L Num_singvalues localD Maxiter verbose


#Expression of variables:

# npes: number of parallel/distributed processors
# Matrix V should be written in a row col value format. Each line corresponds to one element in V.
# Matrix D should be written in a general one line per row matrix format.
# Nf: number of files V is storred in, e.g., 1,2,.... If V is large, it can be storred in multiple files. The V file names should end  in _0, _1, _2, ..._Nf.
# M: number of rows in A (or D)
# N: number of columns in A (or V)
# L: number of columns in D
# Num_singvalues: Number of singular values to be computed
# localD: 0 if a only central node does computation w.r.t to D, set to 1 otherwise
# Maxiter: maximum number of iteraions the power method is run for each singular value (e.g., 50,100,200 ets)
# Verbose:  set to 1 to see reports, set to 0 otherwise. 






