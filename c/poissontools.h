#ifndef POISSONTOOLS_H_
#define POISSONTOOLS_H_

// tools for working with linear system from FEM on unstructured triangulation
// to solve Poisson equation

// preallocate an already-created matrix A
PetscErrorCode prealloc(MPI_Comm comm, Vec E, Vec x, Vec y, Mat A);
#endif
