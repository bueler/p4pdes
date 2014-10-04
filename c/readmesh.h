#ifndef READMESH_H_
#define READMESH_H_

// these methods work with .petsc binary files written by  c2triangle

// get input file name from option "-f", and create corresponding viewer
PetscErrorCode getmeshfile(MPI_Comm comm, const char suffix[], char filename[], PetscViewer *viewer);

// read mesh info from viewers created with getmeshfile()
PetscErrorCode readmesh(MPI_Comm comm, PetscViewer Eviewer, PetscViewer Nviewer,
                        Vec *E,         // length 12*K array with full element info
                        Vec *x, Vec *y, // length N arrays with node coords
                        Vec *Q);        // length 2*M array with node indices
                                        //   for boundary segments

// get sizes: N = (# of nodes), K = (# of elements), M = (# of boundary segments)
PetscErrorCode getmeshsizes(MPI_Comm comm, Vec E, Vec x, Vec Q,
                            PetscInt *N, PetscInt *K, PetscInt *M);
#endif

