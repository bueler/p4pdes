
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n"
"Solves PDE  - div( a(x,y,u) grad u ) = f(x,y,u)  on arbitrary 2D polygonal\n"
"domain, with Dirichlet data g_D(x,y) on portion of boundary.\n"
"Functions a(), f(), and g_D() are given as formulas.\n"
"Input files in PETSc binary format contain node coordinates, elements, and\n"
"boundary flags stored in files.  Allows arbitrary non-homogeneous Dirichlet\n"
"and Neumann conditions along parts of boundary.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_view -un_mesh foo.dat

#include <petsc.h>
#include "um.h"

typedef struct {
    UM  mesh;
    Vec gD;
} unfemCtx;

double a_fcn(double u, double x, double y) {
    return 1.0;
}

double f_fcn(double u, double x, double y) {
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
    ierr = UMAssertValid(&(ctx->mesh)); CHKERRQ(ierr);
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
    ierr = UMAssertValid(&(ctx->mesh)); CHKERRQ(ierr);
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

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[3], double xi, double eta) {
    double sum = 0.0;
    int    q;
    for (q=0; q<3; q++)
        sum += v[q] * chi(q,xi,eta);
    return sum;
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

typedef struct {
    double  x, y;
} grad;

//FIXME should evaluate w.r.t. (x,y)
// evaluate partial derivs of v(xi,eta) on reference element
grad gradeval(const double v[3], double xi, double eta) {
    gradRef tmp;
    grad    sum = {0.0,0.0};
    int     q;
    for (q=0; q<3; q++) {
        tmp = dchi(q,xi,eta);
        // FIXME in here
        sum.x += v[q] * tmp.xi;
        sum.y += v[q] * tmp.eta;
    }
    return sum;
}

double GradInnerProd(grad du, grad dv) {
    return du.x * dv.x + du.y * dv.y;
}

double FunIntegrand(int ell, const double u[3],
                    double a, double f, double xi, double eta) {
  grad dpsi;
  //FIXME: dpsi from  dchi(ell,xi,eta)
  dpsi.x = 0.0;
  dpsi.y = 0.0;
  return a * GradInnerProd(gradeval(u,xi,eta),dpsi) - f * chi(ell,xi,eta);
}

PetscErrorCode FormFunction(SNES snes, Vec u, Vec F, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx     *user = (unfemCtx*)ctx;
    const int    Q = 3; // number of quadrature points
    // quadrature points and weights from bottom page 7 of Shaodeng notes
    const double xi[3]  = {1.0/6.0, 2.0/3.0, 1.0/6.0},
                 eta[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0},
                 w[3]   = {1.0/6.0, 1.0/6.0, 1.0/6.0};
    double       *aF, unode[3],
                 uquad[Q], aquad[Q], fquad[Q],
                 dx1, dx2, dy1, dy2, rho, xx, yy, sum;
    int          n, k, l, q;
    const int    *abf, *ae, *en;
    const double *ax, *ay, *agD, *au;

    ierr = VecSet(F,0.0); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    // Dirichlet node residuals
    ierr = ISGetIndices(user->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->gD,&agD); CHKERRQ(ierr);
    for (n = 0; n < user->mesh.N; n++) {
        if (abf[n] == 2)
            aF[n] = au[n] - agD[n];
    }
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(user->gD,&agD); CHKERRQ(ierr);
    // add element contributions to residuals
    ierr = ISGetIndices(user->mesh.e,&ae); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.y,&ay); CHKERRQ(ierr);
    for (k = 0; k < user->mesh.K; k++) {
        en = ae + 3*k;        // en[0], en[1], en[2] are nodes of element k
        ierr = PetscPrintf(PETSC_COMM_WORLD,"element k=%3d:  en[0]=%d, en[1]=%d, en[2]=%d\n",
                           k, en[0], en[1], en[2]); CHKERRQ(ierr);
        // get geometry of element
        dx1 = ax[en[1]] - ax[en[0]];
        dx2 = ax[en[2]] - ax[en[0]];
        dy1 = ay[en[1]] - ay[en[0]];
        dy2 = ay[en[2]] - ay[en[0]];
        rho = fabs(dx1 * dy2 - dx2 * dy1);
        // get all function values at quadrature nodes on element
        ierr = GetUorG(u,k,unode,user); CHKERRQ(ierr);
        for (q = 0; q < Q; q++) {
            uquad[q] = eval(unode,xi[q],eta[q]);
            xx = ax[en[0]] + dx1 * xi[q] + dx2 * eta[q];
            yy = ay[en[0]] + dy1 * xi[q] + dy2 * eta[q];
            aquad[q] = a_fcn(uquad[q],xx,yy);
            fquad[q] = f_fcn(uquad[q],xx,yy);
        }
        // residual contribution for each node of element
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] < 2) { // if NOT a Dirichlet node
                sum = 0.0;
                for (q = 0; q < Q; q++) {
                    sum += w[q] * FunIntegrand(l,unode,aquad[q],fquad[q],
                                               xi[q],eta[q]);
                }
                aF[en[l]] += rho * sum;
            }
        }
    }
    ierr = ISRestoreIndices(user->mesh.e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(user->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(user->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscBool   view = PETSC_FALSE;
    char        meshroot[256] = "";
    unfemCtx    user;
    SNES        snes;
    Vec         r, u, uexact;
    double      err;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view",
           "view loaded nodes and elements at stdout",
           "unfem.c",view,&view,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh (files have .node,.ele extensions)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // initialize and read mesh/context object of type UM
    ierr = UMInitialize(&(user.mesh)); CHKERRQ(ierr);
    ierr = UMReadNodes(&(user.mesh),meshroot); CHKERRQ(ierr);
    ierr = UMReadElements(&(user.mesh),meshroot); CHKERRQ(ierr);

    // fill fields
    ierr = UMCreateGlobalVec(&(user.mesh),&uexact); CHKERRQ(ierr);
    ierr = FillExact(uexact,&user); CHKERRQ(ierr);
    ierr = VecDuplicate(uexact,&(user.gD)); CHKERRQ(ierr);
    ierr = FillDirichletFromExact(uexact,user.gD,&user); CHKERRQ(ierr);
    if (view) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UMView(&(user.mesh),stdoutviewer); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)(user.gD),"gD");
        ierr = VecView(user.gD,stdoutviewer); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)uexact,"uexact");
        ierr = VecView(uexact,stdoutviewer); CHKERRQ(ierr);
    }

    // configure SNES
    ierr = VecDuplicate(uexact,&r); CHKERRQ(ierr);
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    // solve
    ierr = VecDuplicate(uexact,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

    // measure error
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"result:  |u-u_exact|_inf = %g\n",
                       err); CHKERRQ(ierr);

    // clean-up
    VecDestroy(&(user.gD));  VecDestroy(&uexact);
    VecDestroy(&u);  VecDestroy(&r);
    UMDestroy(&(user.mesh));
    PetscFinalize();
    return 0;
}

