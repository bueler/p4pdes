#ifndef READMESH_H_
#define READMESH_H_

// These utilities are for .petsc binary files which describe planar triangular
// meshes which have either Dirichlet or Neumann boundary.  Such files are
// written by  c2convert,  which converts from output of triangle.

//STARTSTRUCT
typedef struct {
  PetscScalar j[3],  // global indices of vertices (nodes) j[0], j[1], j[2]
              bN[3], // boundary type of node:  bN[0], bN[1], bN[2] in {0,1,2}
              bE[3], // boundary type of edge:  bE[0], bE[1], bE[2] in {0,1,2},
                     //   where bE[0] = <0,1>, bE[1] = <1,2>, bE[2] = <2,0>
              x[3],  // node x-coordinates x[0], x[1], x[2]
              y[3];  // node y-coordinates y[0], y[1], y[2]
} elementtype;
//ENDSTRUCT

// get input file name from option "-f", and create corresponding viewer
PetscErrorCode getmeshfile(MPI_Comm comm, const char suffix[], char filename[], PetscViewer *viewer);

// read mesh info from viewer created with getmeshfile()
PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer,
                        Vec *E,          // length 15*K; has full element info
                        Vec *x, Vec *y); // length N; has node coords for (e.g.)
                                         //   plotting

// get sizes: N = (# of nodes), K = (# of elements)
PetscErrorCode getmeshsizes(MPI_Comm comm, Vec E, Vec x, Vec y,
                            PetscInt *N, PetscInt *K);

// method which is better than VecView for showing a vector which has blocks
//   of type elementtype above
PetscErrorCode elementVecViewSTDOUT(MPI_Comm comm, Vec E);
#endif

