#ifndef UM_H_
#define UM_H_

// data type and functions for unstructured mesh

typedef struct {
    int      N,     // number of nodes
             K,     // number of elements
             PS;    // number of boundary segments
    IS       e,     // element triples; length K
                    //     if ISGetIndices() gets array ae[] then for
                    //     k=0,...,K-1 the values ae[3*k+0],ae[3*k+1],ae[3*k+2]
                    //     are indices into node-based Vecs (in 0,...,N-1)
             bfn,   // flag for boundary node; length N
                    //     if abfn[i] > 0 then node (x_i,y_i) is on boundary
                    //     if abfn[i] == 2 then node (x_i,y_i) is on Dirichlet boundary
             s,     // boundary segment pairs; length PS
                    //     if ISGetIndices() gets array as[] then for
                    //     p=0,...,PS-1 the values as[2*p+0],as[2*p+1]
                    //     are indices into node-based Vecs (in 0,...,N-1)
             bfs;   // flag for boundary segment; length PS
                    //     if abfs[i] == 1 then segment i is on Neumann boundary
                    //     if abfs[i] == 2 then segment i is on Dirichlet boundary
    Vec      x,     // x-coordinate of node; length N
             y;     // y-coordinate of node; length N
} UM;

PetscErrorCode UMInitialize(UM *mesh);
PetscErrorCode UMDestroy(UM *mesh);

// view node coordinates, element triples, boundary flags to STDOUT
PetscErrorCode UMView(UM *mesh, PetscViewer viewer);

// read node coordinates from file and create all nodal-based Vecs
PetscErrorCode UMReadNodes(UM *mesh, char *rootname);

// read element index triples, boundary segments, and boundary node/segment flags,
// creating as IS, from file; call UMReadNodes() first
PetscErrorCode UMReadElements(UM *mesh, char *rootname);
#endif

