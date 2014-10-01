#ifndef READMESH_H_
#define READMESH_H_

// these methods work with .petsc binary files written by  c2triangle

// get input file name from option "-f", and create corresponding viewer
PetscErrorCode getmeshfile(MPI_Comm comm, char filename[], PetscViewer *viewer);

// create a Vec and load it from a PETSc binary file
PetscErrorCode createload(MPI_Comm comm, PetscViewer viewer, Vec *X);

// read all mesh info onto each rank from viewer created with getmeshfile()
PetscErrorCode readmeshseqall(MPI_Comm comm, PetscViewer viewer,
                              Vec *x, Vec *y, // length N arrays with node coords
                              Vec *BT,        // length N array with boundary type:
                                              //   0 = interior,
                                              //   2 = Dirichlet, 3 = Neumann
                              Vec *P,         // length 3*K array with node indices
                                              //   for elements
                              Vec *Q);        // length 2*M array with node indices
                                              //   for boundary segments

// get sizes: N = (# of nodes), K = (# of elements), M = (# of boundary segments)
PetscErrorCode getmeshsizes(MPI_Comm comm, Vec x, Vec P, Vec Q,
                            PetscInt *N, PetscInt *K, PetscInt *M);
#endif

