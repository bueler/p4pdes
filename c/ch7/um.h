#ifndef UM_H_
#define UM_H_

// location of node
typedef struct {
    double   x,y;
} Node;

// data type and functions for unstructured mesh
//STARTUM
typedef struct {
    int      N,     // number of nodes
             K,     // number of elements
             P;     // number of boundary segments
    Vec      loc;   // location of node; length N, dof=2 Vec, with Node entries
    IS       e,     // element triples; length 3K
                    //     values e[3*k+0],e[3*k+1],e[3*k+2]
                    //     are indices into node-based Vecs
             bfn,   // flag for boundary node; length N
                    //     if abfn[i] > 0 then node i is on boundary
                    //     if abfn[i] == 2 then node i is on Dirichlet bdry
             s,     // boundary segment pairs; length 2P
                    //     values s[2*p+0],s[2*p+1]
                    //     are indices into node-based Vecs
             bfs;   // flag for boundary segment; length P
                    //     if abfs[p] == 1 then segment p is on Neumann bdry
                    //     if abfs[p] == 2 then segment p is on Dirichlet bdry
} UM;
//ENDUM

//"methods" below are listed in typical call order

PetscErrorCode UMInitialize(UM *mesh);

// read node coordinates from file; create loc Vec
PetscErrorCode UMReadNodes(UM *mesh, char *rootname);

// read element triples, boundary segments, and boundary node/segment flags
// each created as an IS
// call UMReadNodes() first
PetscErrorCode UMReadISs(UM *mesh, char *rootname);

// view node coordinates, element triples, boundary segments, and boundary node/segment flags
PetscErrorCode UMView(UM *mesh, PetscViewer viewer);

PetscErrorCode UMDestroy(UM *mesh);
#endif

