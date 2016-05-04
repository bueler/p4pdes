#ifndef UF_H_
#define UF_H_

// Abstract data type (object) for holding unstructured mesh.

typedef struct {
    int      N,     // number of nodes
             K;     // number of elements
    IS       e,     // element triples; length K
                    //     if ISGetIndices() gets array ae[] then
                    //     for k=0,...,K-1 the values ae[3*k+0],ae[3*k+1],ae[3*k+2]
                    //     are indices into node-based Vecs (in 0,...,N-1)
             bf;    // boundary flag; length N
                    //     if bf[i] > 0 then (x_i,y_i) is on boundary
                    //     if bf[i] == 2 then (x_i,y_i) is on Dirichlet boundary
    // length N Vecs:
    Vec      x,     // x-coordinate of node
             y;     // y-coordinate of node
} UF;


PetscErrorCode UFInitialize(UF *mesh);
PetscErrorCode UFDestroy(UF *mesh);

PetscErrorCode UFView(UF *mesh, PetscViewer viewer);

// read node coordinates from file and create all nodal-based Vecs
PetscErrorCode UFReadNodes(UF *mesh, char *rootname);

// read element index triples and boundary flags, creating as IS, from file
PetscErrorCode UFReadElements(UF *mesh, char *rootname);

// check element triples for admissibility
PetscErrorCode UFCheckElements(UF *mesh);

// check boundary flags for admissibility
PetscErrorCode UFCheckBoundaryFlags(UF *mesh);

PetscErrorCode UFAssertValid(UF *mesh);

PetscErrorCode UFCreateGlobalVec(UF *mesh, Vec *v);
#endif

