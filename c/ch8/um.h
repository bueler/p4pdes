#ifndef UM_H_
#define UM_H_

// abstractish data type (object) for holding unstructured mesh

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
} UM;

PetscErrorCode UMInitialize(UM *mesh);
PetscErrorCode UMDestroy(UM *mesh);

// view node coordinates, element triples, boundary flags to STDOUT
PetscErrorCode UMView(UM *mesh, PetscViewer viewer);

// read node coordinates from file and create all nodal-based Vecs
PetscErrorCode UMReadNodes(UM *mesh, char *rootname);

// read element index triples and boundary flags, creating as IS, from file
// call UMReadNodes() firsts
PetscErrorCode UMReadElements(UM *mesh, char *rootname);

// allocate Vec v with size equal to N = number of nodes
PetscErrorCode UMCreateGlobalVec(UM *mesh, Vec *v);
#endif

