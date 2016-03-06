project(GraphLab)
# include(CheckCXXSourceCompiles)

# Build gd
add_graphlab_executable(gd gd.cpp)
requires_eigen(gd) # build and attach eigen
