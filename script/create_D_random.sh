#!/bin/bash

# This code performssubspace sampling on a matrix A. The output is a dictionary D, whose columns are selected randomly from columns of A, such that ||A-DD+A||_F < epsilon*||A||_F


#Notes for comiling and executing:

#For compiling modules GCC, OpenMPI (or alternatives) should be loaded, and the EIGEN library should be installed


#For compiling the code:
mpic++  -o outpuname.o ../src/create_D_random.cpp

#For running the code:
mpiexec -n npes outpuname.o "A directory" "desired D directory" M N Lmin Lstep lmax epsilon verbose 



#Definition of variables:

# Matrix A should be written in a general one line per row matrix format.
# M: number of rows in A (or D)
# N: number of columns in A (or V)
# Lmin: starting number of randomly selected columns to create D
# Lstep: batch of columns to be added to D before a norm-2 error evaluation is performed to check if current D meets the error criteria.
# Lmax: maximum number of columns in D
# epsilon: norm-2 error such that ||A-DD^+A||_F < epsilon*||A||_F (0<epsilon<=1). D^+ is the psudoinverse of D.
# verbose:  set to 1 to see reports, set to 0 otherwise. 


# The output D is written in  "desired D directory" in a general one line per row matrix format.

