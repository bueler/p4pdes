#ifndef POISSONTOOLS_H_
#define POISSONTOOLS_H_

// tools for working with linear system from FEM on unstructured triangulation
// to solve Poisson equation

// preallocate an already-created matrix A
PetscErrorCode prealloc(MPI_Comm comm, Vec E, Vec x, Vec y, Mat A);

// initial assembly of A, b (A, b must already be created; result should be correct
//   for Neumann-only case; ignores preallocation; ignors Dirichlet g)
PetscErrorCode initassemble(MPI_Comm comm,
                            Vec E,         // array of elementtype, as read by readmesh()
                            PetscScalar (*f)(PetscScalar, PetscScalar),
                            PetscScalar (*gamma)(PetscScalar, PetscScalar),
                            Mat A, Vec b);
#endif
