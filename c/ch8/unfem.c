
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n"
"Solves PDE  - div( a(x,y,u) grad u ) = f(x,y,u)  on arbitrary 2D polygonal\n"
"domain, with Dirichlet data g_D(x,y) on portion of boundary.\n"
"Functions a(), f(), and g_D() are given as formulas.\n"
"Input files in PETSc binary format contain node coordinates, elements, and\n"
"boundary flags stored in files.  Allows non-homogeneous Dirichlet\n"
"and Neumann conditions along subsets of boundary.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_view -un_mesh foo.dat -snes_fd

// with view of mat:
//   $ ./unfem -un_mesh foo.dat -snes_fd -snes_monitor -ksp_monitor -mat_view draw -draw_pause 1

#include <petsc.h>
#include "um.h"

typedef struct {
    UM  mesh;
    Vec gD;
    //Vec gN;  <--- FIXME TODO
} unfemCtx;

double a_fcn(double u, double x, double y) {
    return 1.0;
}

double f_fcn(double u, double x, double y) {
    return 2.0 + 3.0 * y * y;
}

double uexact_fcn(double x, double y) {
    const double y2 = y * y;
    return 1.0 - y2 - 0.25 * y2 * y2;
}

PetscErrorCode FillExactAndDirichlet(Vec uexact, Vec gD, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const double *ax, *ay;
    const int    *abf;
    double       *auexact, *agD;
    int          i;
    ierr = ISGetIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(uexact,&auexact); CHKERRQ(ierr);
    ierr = VecGetArray(gD,&agD); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh.N; i++) {
        auexact[i] = uexact_fcn(ax[i],ay[i]);
        agD[i] = (abf[i] == 2) ? auexact[i] : NAN;
    }
    ierr = VecRestoreArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(uexact,&auexact); CHKERRQ(ierr);
    ierr = VecRestoreArray(gD,&agD); CHKERRQ(ierr);
    ierr = ISRestoreIndices(ctx->mesh.bf,&abf); CHKERRQ(ierr);
    return 0;
}

double chi(int L, double xi, double eta) {
    if (L == 0) {
        return 1.0 - xi - eta;
    } else if (L == 1) {
        return xi;
    } else {
        return eta;
    }
}

const double dchi[3][2] = {{-1.0,-1.0},
                           { 1.0, 0.0},
                           { 0.0, 1.0}};

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[3], double xi, double eta) {
    double sum = 0.0;
    int    L;
    for (L = 0; L < 3; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

double InnerProd(const double V[2], const double W[2]) {
    return V[0] * W[0] + V[1] * W[1];
}

// quadrature points and weights from bottom page 7 of Shaodeng notes
const int    Q = 3; // number of quadrature points
const double xi[3]  = {1.0/6.0, 2.0/3.0, 1.0/6.0},
             eta[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0},
             w[3]   = {1.0/6.0, 1.0/6.0, 1.0/6.0};

PetscErrorCode FormFunction(SNES snes, Vec u, Vec F, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx     *user = (unfemCtx*)ctx;
    double       *aF, unode[3], gradu[2], gradpsi[3][2],
                 uquad[Q], aquad[Q], fquad[Q],
                 dx1, dx2, dy1, dy2, detJ, xx, yy, sum;
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
    // add element contributions to residuals
    ierr = ISGetIndices(user->mesh.e,&ae); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.y,&ay); CHKERRQ(ierr);
    for (k = 0; k < user->mesh.K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        // geometry of element
        dx1 = ax[en[1]] - ax[en[0]];
        dx2 = ax[en[2]] - ax[en[0]];
        dy1 = ay[en[1]] - ay[en[0]];
        dy2 = ay[en[2]] - ay[en[0]];
        detJ = dx1 * dy2 - dx2 * dy1;
        // gradients of hat functions
        for (l = 0; l < 3; l++) {
            gradpsi[l][0] = ( dy2 * dchi[l][0] - dy1 * dchi[l][1]) / detJ;
            gradpsi[l][1] = (-dx2 * dchi[l][0] + dx1 * dchi[l][1]) / detJ;
        }
        // u and grad u on element
        gradu[0] = 0.0;
        gradu[1] = 0.0;
        for (l = 0; l < 3; l++) {
            unode[l] = (abf[en[l]] == 2) ? agD[en[l]] : au[en[l]];
            gradu[0] += unode[l] * gradpsi[l][0];
            gradu[1] += unode[l] * gradpsi[l][1];
        }
        // function values at quadrature points on element
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
                    sum += w[q] * ( aquad[q] * InnerProd(gradu,gradpsi[l])
                                    - fquad[q] * chi(l,xi[q],eta[q]) );
                }
                aF[en[l]] += fabs(detJ) * sum;
            }
        }
    }
    ierr = ISRestoreIndices(user->mesh.e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(user->gD,&agD); CHKERRQ(ierr);
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

    // read mesh/context object of type UM
    ierr = UMInitialize(&(user.mesh)); CHKERRQ(ierr);
    ierr = UMReadNodes(&(user.mesh),meshroot); CHKERRQ(ierr);
    ierr = UMReadElements(&(user.mesh),meshroot); CHKERRQ(ierr);
    if (view) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UMView(&(user.mesh),stdoutviewer); CHKERRQ(ierr);
    }

    // fill fields: boundary values and exact solution
    ierr = UMCreateGlobalVec(&(user.mesh),&uexact); CHKERRQ(ierr);
    ierr = VecDuplicate(uexact,&(user.gD)); CHKERRQ(ierr);
    ierr = FillExactAndDirichlet(uexact,user.gD,&user); CHKERRQ(ierr);

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
    SNESDestroy(&snes);
    VecDestroy(&(user.gD));  VecDestroy(&uexact);
    VecDestroy(&u);  VecDestroy(&r);
    UMDestroy(&(user.mesh));
    PetscFinalize();
    return 0;
}

