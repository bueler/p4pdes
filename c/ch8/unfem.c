
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_check -un_mesh foo.dat

#include <petsc.h>

typedef struct {
    int      N,     // number of nodes
             K;     // number of elements
    Vec      x, y;  // coordinates of nodes; length N
    IS       e;     // element triples;  if ISGetIndices() gets array ae[] then
                    //     for k=0,...,K-1 the values ae[3*k+0],ae[3*k+1],ae[3*k+2]
                    //     are indices into node-based Vecs (in 0,...,N-1)
    Vec      u,     // approximate solution at nodes; length N
             uexact;// exact solution at nodes; length N
} UF;

PetscErrorCode UFInitialize(UF *ctx) {
    ctx->N = 0;
    ctx->K = 0;
    return 0;
}

//FIXME add Viewer argument?
PetscErrorCode UFView(UF *ctx) {
    PetscErrorCode ierr;
    const double *ax, *ay;
    int          n, k;
    const int    *ae;
    if (ctx->N > 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%d nodes:\n",ctx->N); CHKERRQ(ierr);
        ierr = VecGetArrayRead(ctx->x,&ax); CHKERRQ(ierr);
        ierr = VecGetArrayRead(ctx->y,&ay); CHKERRQ(ierr);
        for (n = 0; n < ctx->N; n++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d : (%g,%g)\n",
                               n,ax[n],ay[n]); CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(ctx->x,&ax); CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(ctx->y,&ay); CHKERRQ(ierr);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"nodes empty\n"); CHKERRQ(ierr);
    }
    if (ctx->K > 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%d elements:\n",ctx->K); CHKERRQ(ierr);
        ierr = ISGetIndices(ctx->e,&ae); CHKERRQ(ierr);
        for (k = 0; k < ctx->K; k++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d : %3d %3d %3d\n",
                               k,ae[3*k+0],ae[3*k+1],ae[3*k+2]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(ctx->e,&ae); CHKERRQ(ierr);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"elements empty\n"); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode UFReadNodes(UF *ctx, char *filename) {
    PetscErrorCode ierr;
    int         m;
    PetscViewer viewer;
    if (ctx->N > 0) {
        SETERRQ(PETSC_COMM_WORLD,1,"nodes already created?\n");
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&ctx->x); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&ctx->y); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ctx->x); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ctx->y); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(ctx->x,viewer); CHKERRQ(ierr);
    ierr = VecLoad(ctx->y,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(ctx->x,&(ctx->N)); CHKERRQ(ierr);
    ierr = VecGetSize(ctx->y,&m); CHKERRQ(ierr);
    if (ctx->N != m) {
        SETERRQ1(PETSC_COMM_WORLD,2,"node coordinates x,y loaded from %s are not the same size\n",filename);
    }
    return 0;
}


PetscErrorCode UFReadElements(UF *ctx, char *filename) {
    PetscErrorCode ierr;
    PetscViewer viewer;
    const int   *ae;
    int         k, m;
    // create IS and load
    if (ctx->K > 0) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "elements already created? ... stopping\n");
    }
    ierr = ISCreate(PETSC_COMM_WORLD,&(ctx->e)); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = ISLoad(ctx->e,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = ISGetSize(ctx->e,&(ctx->K)); CHKERRQ(ierr);
    if (ctx->K % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,2,
                 "IS e loaded from %s is wrong size for list of element triples\n",filename);
    }
    ctx->K /= 3;
    // check element triples for admissibility
    if (ctx->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "node size unknown so element check impossible; call UFReadNodes() first\n");
    }
    ierr = ISGetIndices(ctx->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < ctx->K; k++) {
        for (m = 0; m < 3; m++) {
            if ((ae[3*k+m] < 0) || (ae[3*k+m] >= ctx->N)) {
                SETERRQ3(PETSC_COMM_WORLD,2,
                   "index e[%d]=%d invalid: not between 0 and N-1=%d\n",
                   3*k+m,ae[3*k+m],ctx->N-1);
            }
        }
        // FIXME: could add check of distinct indices
    }
    ierr = ISRestoreIndices(ctx->e,&ae); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UFDestroy(UF *ctx) {
    ISDestroy(&(ctx->e));
    VecDestroy(&(ctx->x));
    VecDestroy(&(ctx->y));
    VecDestroy(&(ctx->uexact));
    return 0;
}

PetscErrorCode UFFillExact(UF *ctx) {
    PetscErrorCode ierr;
    const double *ay;
    double       y, *auexact;
    int          i;
    if (ctx->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,1,"nodes must exist before exact solution can be calculated\n");
    }
    ierr = VecDuplicate(ctx->y,&ctx->uexact); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->y,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(ctx->uexact,&auexact); CHKERRQ(ierr);
    for (i = 0; i < ctx->N; i++) {
        y = ay[i];
        auexact[i] = 1.0 - y*y - y*y*y*y;
    }
    ierr = VecRestoreArrayRead(ctx->y,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(ctx->uexact,&auexact); CHKERRQ(ierr);
    return 0;
}


#define DEBUG_REFERENCE_EVAL 1

double chi(int L, double xi, double eta) {
#ifdef DEBUG_REFERENCE_EVAL
    if ((xi < 0.0) || (xi > 1.0) || (eta < 0.0) || (eta > 1.0 - xi)) {
        PetscPrintf(PETSC_COMM_WORLD,"chi(): coordinates (xi,eta) outside of reference element\n");
        PetscEnd();
    }
#endif
    switch (L) {
        case 1 :
            return xi;
        case 2 :
            return eta;
        default :
            return 1.0 - xi - eta;
    }
}

typedef struct {
    double  xi, eta;
} gradRef;

gradRef dchi(int q, double xi, double eta) {
#ifdef DEBUG_REFERENCE_EVAL
    if ((xi < 0.0) || (xi > 1.0) || (eta < 0.0) || (eta > 1.0 - xi)) {
        PetscPrintf(PETSC_COMM_WORLD,"chi(): coordinates (xi,eta) outside of reference element\n");
        PetscEnd();
    }
#endif
    switch (q) {
        case 1 :
            return (gradRef){1.0, 0.0};
        case 2 :
            return (gradRef){0.0, 1.0};
        default :
            return (gradRef){-1.0, -1.0};
    }
}

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[3], double xi, double eta) {
    double sum = 0.0;
    int    q;
    for (q=0; q<3; q++)
        sum += v[q] * chi(q,xi,eta);
    return sum;
}

// evaluate partial derivs of v(xi,eta) on reference element
gradRef deval(const double v[3], double xi, double eta) {
    gradRef sum = {0.0,0.0}, tmp;
    int     q;
    for (q=0; q<3; q++) {
        tmp = dchi(q,xi,eta);
        sum.xi  += v[q] * tmp.xi;
        sum.eta += v[q] * tmp.eta;
    }
    return sum;
}

double a_eval(double x, double y, double u) {
    return 1.0;
}

double f_eval(double x, double y, double u) {
    return -2.0 + 12.0 * y * y;
}

double GradInnerProd(gradRef du, gradRef dv) {
    //FIXME return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
    return 0.0;
}

double FunIntegrand(int q, const double u[3],
                    double a, double f, double xi, double eta) {
  const gradRef du    = deval(u,xi,eta),
                dchiq = dchi(q,xi,eta);
  return a * GradInnerProd(du,dchiq) - f * chi(q,xi,eta);
}


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscBool   check = PETSC_FALSE;
    char        meshroot[256] = "", nodename[266], elename[266];
    UF          mesh;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-check",
           "check on loaded nodes and elements",
           "unfem.c",check,&check,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh (files have .node,.ele extensions)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    strcpy(nodename, meshroot);
    strncat(nodename, ".node", 10);
    strcpy(elename, meshroot);
    strncat(elename, ".ele", 10);

    ierr = UFInitialize(&mesh); CHKERRQ(ierr);
    ierr = UFReadNodes(&mesh,nodename); CHKERRQ(ierr);
    ierr = UFReadElements(&mesh,elename); CHKERRQ(ierr);
    if (check) {
        ierr = UFView(&mesh); CHKERRQ(ierr);
    }
    ierr = UFFillExact(&mesh); CHKERRQ(ierr);

    UFDestroy(&mesh);
    PetscFinalize();
    return 0;
}

