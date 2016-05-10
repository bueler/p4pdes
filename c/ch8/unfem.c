
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n"
"Solves PDE  - div( a(u,x,y) grad u ) = f(u,x,y)  on arbitrary 2D polygonal\n"
"domain, with Dirichlet data g_D(x,y) on portion of boundary.\n"
"Functions a(), f(), g_D(), and u_exact() are given as formulas.\n"
"(There are three different solution cases implemented for these functions.)\n"
"Input files in PETSc binary format contain node coordinates, elements, and\n"
"boundary flags stored in files.  Allows non-homogeneous Dirichlet\n"
"and Neumann conditions along subsets of boundary.\n\n";

// example:
//   $ ./tri2petsc.py meshes/trap.1 trap.1.dat
//   $ ./unfem -un_view -un_mesh trap.1.dat -snes_fd

// with view of mat:
//   $ ./unfem -un_mesh trap.1.dat -snes_fd -snes_monitor -ksp_monitor -mat_view draw -draw_pause 1

#include <petsc.h>
#include "um.h"
#include "solutioncases.h"

typedef struct {
    UM     mesh;
    int    solncase,
           quaddeg;
    double (*a_fcn)(double, double, double);
    double (*f_fcn)(double, double, double);
    double (*gD_fcn)(double, double);
    //double (*gN_fcn)(double, double);  <--- FIXME TODO
    double (*uexact_fcn)(double, double);
} unfemCtx;

PetscErrorCode FillExact(Vec uexact, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const double *ax, *ay;
    double       *auexact;
    int          i;
    ierr = VecGetArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(uexact,&auexact); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh.N; i++) {
        auexact[i] = ctx->uexact_fcn(ax[i],ay[i]);
    }
    ierr = VecRestoreArrayRead(ctx->mesh.y,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(uexact,&auexact); CHKERRQ(ierr);
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

// quadrature points and weights from Shaodeng notes
const int    Q[3] = {1, 3, 4};
const double w[3][4] = {{1.0/2.0, NAN, NAN, NAN},
                        {1.0/6.0, 1.0/6.0, 1.0/6.0, NAN},
                        {-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0}},
             xi[3][4]  = {{1.0/3.0, NAN, NAN, NAN},
                          {1.0/6.0, 2.0/3.0, 1.0/6.0, NAN},
                          {1.0/3.0, 1.0/5.0, 3.0/5.0, 1.0/5.0}},
             eta[3][4] = {{1.0/3.0, NAN, NAN, NAN},
                          {1.0/6.0, 1.0/6.0, 2.0/3.0, NAN},
                          {1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0}};

PetscErrorCode FormFunction(SNES snes, Vec u, Vec F, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx     *user = (unfemCtx*)ctx;
    const int    *abf, *ae, *en, deg = user->quaddeg - 1;
    const double *ax, *ay, *au;
    double       *aF, unode[3], gradu[2], gradpsi[3][2],
                 uquad[4], aquad[4], fquad[4],
                 dx1, dx2, dy1, dy2, detJ, xx, yy, sum;
    int          n, k, l, q;

    ierr = VecSet(F,0.0); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    // Dirichlet node residuals
    ierr = ISGetIndices(user->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.x,&ax); CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->mesh.y,&ay); CHKERRQ(ierr);
    for (n = 0; n < user->mesh.N; n++) {
        if (abf[n] == 2)
            aF[n] = au[n] - user->gD_fcn(ax[n],ay[n]);
    }
    // add element contributions to residuals
    ierr = ISGetIndices(user->mesh.e,&ae); CHKERRQ(ierr);
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
            unode[l] = (abf[en[l]] == 2) ? user->gD_fcn(ax[en[l]],ay[en[l]]) : au[en[l]];
            gradu[0] += unode[l] * gradpsi[l][0];
            gradu[1] += unode[l] * gradpsi[l][1];
        }
        // function values at quadrature points on element
        for (q = 0; q < Q[deg]; q++) {
            uquad[q] = eval(unode,xi[deg][q],eta[deg][q]);
            xx = ax[en[0]] + dx1 * xi[deg][q] + dx2 * eta[deg][q];
            yy = ay[en[0]] + dy1 * xi[deg][q] + dy2 * eta[deg][q];
            aquad[q] = user->a_fcn(uquad[q],xx,yy);
            fquad[q] = user->f_fcn(uquad[q],xx,yy);
        }
        // residual contribution for each node of element
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] < 2) { // if NOT a Dirichlet node
                sum = 0.0;
                for (q = 0; q < Q[deg]; q++) {
                    sum += w[deg][q] * ( aquad[q] * InnerProd(gradu,gradpsi[l])
                                         - fquad[q] * chi(l,xi[deg][q],eta[deg][q]) );
                }
                aF[en[l]] += fabs(detJ) * sum;
            }
        }
    }
    ierr = ISRestoreIndices(user->mesh.e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh.bf,&abf); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
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
    user.quaddeg = 2;
    user.solncase = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-case",
           "exact solution cases: 0=linear, 1=nonlinear, 2=chapter3",
           "unfem.c",user.solncase,&(user.solncase),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh (files have .node,.ele extensions)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddeg",
           "quadrature degree (1,2,3)",
           "unfem.c",user.quaddeg,&(user.quaddeg),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view",
           "view loaded nodes and elements at stdout",
           "unfem.c",view,&view,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // set parameters and exact solution
    user.a_fcn = &a_lin;
    user.f_fcn = &f_lin;
    user.uexact_fcn = &uexact_lin;
    user.gD_fcn = &gD_lin;
    switch (user.solncase) {
        case 0 :
            break;
        case 1 :
            user.a_fcn = &a_nonlin;
            user.f_fcn = &f_nonlin;
            break;
        case 2 :
            user.a_fcn = &a_square;
            user.f_fcn = &f_square;
            user.uexact_fcn = &uexact_square;
            user.gD_fcn = &gD_square;
            break;
        default :
            SETERRQ(PETSC_COMM_WORLD,1,"other solution cases not implemented");
    }

    // read mesh object of type UM
    ierr = UMInitialize(&(user.mesh)); CHKERRQ(ierr);
    ierr = UMReadNodes(&(user.mesh),meshroot); CHKERRQ(ierr);
    ierr = UMReadElements(&(user.mesh),meshroot); CHKERRQ(ierr);
    if (view) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UMView(&(user.mesh),stdoutviewer); CHKERRQ(ierr);
    }

    // configure SNES
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = UMCreateGlobalVec(&(user.mesh),&r); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    // set initial iterate and solve
    ierr = VecDuplicate(r,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

    // measure error relative to exact solution
    ierr = VecDuplicate(r,&uexact); CHKERRQ(ierr);
    ierr = FillExact(uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"result for N=%d nodes:  |u-u_exact|_inf = %g\n",
                       user.mesh.N,err); CHKERRQ(ierr);

    // clean-up
    SNESDestroy(&snes);  UMDestroy(&(user.mesh));
    VecDestroy(&u);  VecDestroy(&r);  VecDestroy(&uexact);
    PetscFinalize();
    return 0;
}

