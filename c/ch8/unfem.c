
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_check -un_mesh foo.dat

#include <petsc.h>


typedef struct {
    int node[3];
} Element;

typedef struct {
    int      N,     // number of nodes
             K;     // number of elements
    Vec      x, y;  // coordinates of nodes
    Element  *e;    // e[0],e[1],e[2] are indices into nodes (in 0,...,N-1)
    Vec      u,     // approximate solution
             uexact;// exact solution
} UF;

//FIXME add Viewer argument?
PetscErrorCode UFView(UF *ctx) {
    PetscErrorCode ierr;
    double  *ax, *ay;
    int     n, k;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d nodes:\n",ctx->N); CHKERRQ(ierr);
    ierr = VecGetArray(ctx->x,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(ctx->y,&ay); CHKERRQ(ierr);
    for (n = 0; n < ctx->N; n++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d = (%g,%g)\n",
                           n,ax[n],ay[n]); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(ctx->x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(ctx->y,&ay); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d elements:\n",ctx->K); CHKERRQ(ierr);
    for (k = 0; k < ctx->K; k++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    e[%3d] = %3d %3d %3d\n",
                           k,ctx->e[k].node[0],ctx->e[k].node[1],ctx->e[k].node[2]); CHKERRQ(ierr);
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

//FIXME should use IS for elements
//    IS e;
//    ierr = ISCreate(PETSC_COMM_WORLD,&e); CHKERRQ(ierr);
//    ierr = ISLoad(e,viewer); CHKERRQ(ierr);  <-- FAILS  see petsc issue #127
//    ierr = ISView(e,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//    ISDestroy(&e);

PetscErrorCode UFReadElements(UF *ctx, char *filename) {
    PetscErrorCode ierr;
    double      *ae;
    int         k, m;
    Vec         evec;
    PetscViewer viewer;
    if (ctx->K > 0) {
        SETERRQ(PETSC_COMM_WORLD,1,"elements already created?\n");
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&evec); CHKERRQ(ierr);
    ierr = VecSetFromOptions(evec); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(evec,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(evec,&k); CHKERRQ(ierr);
    if (k % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,2,"evec loaded from %s is wrong size for list of element triples\n",filename);
    }
    ctx->K = k/3;
    if (ctx->e) {
        SETERRQ(PETSC_COMM_WORLD,3,"something wrong ... ctx->e already malloced\n");
    }
    ctx->e = (Element*)malloc(ctx->K * sizeof(Element));
    ierr = VecGetArray(evec,&ae); CHKERRQ(ierr);
    for (k = 0; k < ctx->K; k++) {
        for (m = 0; m < 3; m++)
            ctx->e[k].node[m] = ae[3*k+m];
    }
    ierr = VecRestoreArray(evec,&ae); CHKERRQ(ierr);
    VecDestroy(&evec);
    return 0;
}

PetscErrorCode UFDestroy(UF *ctx) {
    if (ctx->e)
        free(ctx->e);
    VecDestroy(&(ctx->x));
    VecDestroy(&(ctx->y));
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
    mesh.N = 0;
    mesh.K = 0;
    mesh.e = NULL;
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

    ierr = UFReadNodes(&mesh,nodename); CHKERRQ(ierr);
    ierr = UFReadElements(&mesh,elename); CHKERRQ(ierr);
    if (check) {
        ierr = UFView(&mesh); CHKERRQ(ierr);
    }

    UFDestroy(&mesh);
    PetscFinalize();
    return 0;
}

