#!/bin/bash

# This code performs orthogonal matching pursuit on a matrix A, with a given dictionary D, the output is a sparse V such that A ~ DV  
# Where A is approximated by D*V. D is a dictionary matrix and V is a coefficient matric. The factorization can be achieved using the RankMap methodology or other factorization schemes.
# We use the batch OMP with LDLT decompostion approach


#Notes for comiling and executing:

#For compiling modules GCC, OpenMPI (or alternatives) should be loaded, and the EIGEN library should be installed


#For compiling the code:
mpic++  -o outpuname.o ../src/multiomp.cpp

#For running the code:
mpiexec -n npes outpuname.o "A directory" "desired V directory" "D directory" M N L kratio epsilon output verbose

#Definition of variables:

# Matrix A should be written in a general one line per row matrix format.
# Matrix D should be written in a general one line per row matrix format.
# M: number of rows in A (or D)
# N: number of columns in A (or V)
# L: number of columns in D
# kratio: equals K/L, where k is the desired number of non-zeros per column of V (1/L<=kratio<=1). 
# epsilon: OMP normalized error (0<epsilon<=1). 
# output: set to 1 for writing output files in file
# Verbose:  set to 1 to see reports, set to 0 otherwise. 

# if (kratio==1), omp runs until the error is less than epsilon. 
# if error criteria epsilon is met, OMP stops. This is regardles of what kratio is set to.

# The output V files are written in  "V directory"_"x"_L_npes_id files. where id = 0,1,2,...,npes-1 
# Matrix V is written in a row col value format. Each line corresponds to one element in V.



