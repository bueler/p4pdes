
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n"
"Solves PDE  - div( a(x,y,u) grad u ) = f(x,y,u)  on arbitrary 2D polygonal\n"
"domain, with dirichlet data g_D(x,y) on portion of boundary.\n"
"Functions a(), f(), and g_D() are given as formulas.\n"
"Input files in PETSc binary format contain node coordinates, elements, and\n"
"boundary flags stored in files.  Allows arbitrary non-homogeneous Dirichlet\n"
"and Neumann conditions along parts of boundary.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_view -un_mesh foo.dat

#include <petsc.h>
#include "uf.h"

typedef struct {
    UF  mesh;
    Vec gD;
} unfemCtx;

double a_fcn(double x, double y, double u) {
    return 1.0;
}

double f_fcn(double x, double y, double u) {
    return -2.0 + 12.0 * y * y;
}

double uexact_fcn(double x, double y) {
    const double y2 = y * y;
    return 1.0 - y2 - y2 * y2;
}

PetscErrorCode FillExact(Vec uexact, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const double *ax, *ay;
    double       *auexact;
    int          i;
    ierr = UFAssertValid(&(ctx->mesh)); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(uexact,&auexact); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh.N; i++) {
        auexact[i] = uexact_fcn(ax[i],ay[i]);
    }
    ierr = VecRestoreArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(uexact,&auexact); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FillDirichletFromExact(Vec uexact, Vec gD, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const double *auexact;
    const int    *abf;
    double       *agD;
    int          i;
    ierr = UFAssertValid(&(ctx->mesh)); CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecGetArrayRead(uexact,&auexact); CHKERRQ(ierr);
    ierr = VecGetArray(gD,&agD); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh.N; i++) {
        if (abf[i] == 2)
            agD[i] = auexact[i];
        else
            agD[i] = NAN;
    }
    ierr = VecGetArray(gD,&agD); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(uexact,&auexact); CHKERRQ(ierr);
    ierr = ISRestoreIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    return 0;
}


// evaluate u or g, according to whether the node is on
// the Dirichlet boundary or not, at the 3 vertices of triangle k
PetscErrorCode GetUorG(Vec u, int k, double *uvertex, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const int    *ae, *abf;
    const double *au, *agD;
    int          i, m;
    ierr = ISGetIndices(ctx->mesh.e,&ae); CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->gD,&agD); CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    for (m = 0; m < 3; m++) {
        i = ae[3*k+m];   // node index for vertex m of triangle k
        uvertex[m] = (abf[i] == 2) ? agD[i] : au[i];
    }
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->gD,&agD); CHKERRQ(ierr);
    ierr = ISRestoreIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = ISRestoreIndices(ctx->mesh.e,&ae); CHKERRQ(ierr);
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


//FIXME PetscErrorCode FormFunction(Vec u, Vec F, void *ctx)  ?


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscBool   view = PETSC_FALSE;
    char        meshroot[256] = "";
    unfemCtx    user;
    Vec         u, uexact;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view",
           "view loaded nodes and elements at stdout",
           "unfem.c",view,&view,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh (files have .node,.ele extensions)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // initialize and read mesh/context object of type UF
    ierr = UFInitialize(&(user.mesh)); CHKERRQ(ierr);
    ierr = UFReadNodes(&(user.mesh),meshroot); CHKERRQ(ierr);
    ierr = UFReadElements(&(user.mesh),meshroot); CHKERRQ(ierr);
    ierr = UFAssertValid(&(user.mesh)); CHKERRQ(ierr);

    // fill fields
    ierr = UFCreateGlobalVec(&(user.mesh),&uexact); CHKERRQ(ierr);
    ierr = FillExact(uexact,&user); CHKERRQ(ierr);
    ierr = VecDuplicate(uexact,&(user.gD)); CHKERRQ(ierr);
    ierr = FillDirichletFromExact(uexact,user.gD,&user); CHKERRQ(ierr);
    if (view) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UFView(&(user.mesh),stdoutviewer); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)(user.gD),"gD");
        ierr = VecView(user.gD,stdoutviewer); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)uexact,"uexact");
        ierr = VecView(uexact,stdoutviewer); CHKERRQ(ierr);
    }

    // solve
    ierr = VecDuplicate(uexact,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    // SNESSolve() here

    // clean-up
    VecDestroy(&u);  VecDestroy(&(user.gD));  VecDestroy(&uexact);
    UFDestroy(&(user.mesh));
    PetscFinalize();
    return 0;
}

