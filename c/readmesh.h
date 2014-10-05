#ifndef READMESH_H_
#define READMESH_H_

// these methods work with .petsc binary files written by  c2convert

typedef struct {
  PetscScalar j[3],  // global indices of nodes j[0], j[1], j[2]
              BT[3], // boundary type of node:  BT[0], BT[1], BT[2]
              x[3],  // node x-coordinate x[0], x[1], x[2]
              y[3];  // node y-coordinate y[0], y[1], y[2]
} elementtype;

// get input file name from option "-f", and create corresponding viewer
PetscErrorCode getmeshfile(MPI_Comm comm, const char suffix[], char filename[], PetscViewer *viewer);

// read mesh info from viewer created with getmeshfile()
PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer,
                        Vec *E,         // length 12*K array with full element info
                        Vec *x, Vec *y, // length N arrays with node coords
                        Vec *Q);        // length 2*M array with node indices
                                        //   for boundary segments

// get sizes: N = (# of nodes), K = (# of elements), M = (# of boundary segments)
PetscErrorCode getmeshsizes(MPI_Comm comm, Vec E, Vec x, Vec Q,
                            PetscInt *N, PetscInt *K, PetscInt *M);
#endif

