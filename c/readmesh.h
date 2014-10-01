#ifndef READMESH_H_
#define READMESH_H_

// these methods work with .petsc binary files written by  c2triangle

// get input file name from option "-f", and create corresponding viewer
PetscErrorCode getmeshfile(MPI_Comm comm, char filename[], PetscViewer *viewer);

// read from viewer created with getmeshfile()
PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer,
                        PetscInt *N,    // number of nodes
                        PetscInt *K,    // number of elements
                        PetscInt *M,    // number of boundary segments
                        Vec *x, Vec *y, // length N arrays (parallel) with nodes
                        Vec *BTseq,     // length N array (sequential)
                                        //   with boundary type: 0 = interior,
                                        //   2 = Dirichlet, 3 = Neumann
                        Vec *Pseq,      // length 3*K array (sequential)
                                        //   with node indices for elements
                        Vec *Qseq);     // length 2*M array (sequential)
                                        //   with node indices for boundary segments

#endif

