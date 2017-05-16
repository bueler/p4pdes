static char help[] =
"1D Stokes problem with DMDA and SNES.  Option prefix -wel_.\n\n";

// show solution (staggered):
// ./well -snes_monitor -snes_converged_reason -ksp_type preonly -pc_type svd -da_refine 7 -snes_monitor_solution draw -draw_pause 2

// try:
//   -ksp_type preonly -pc_type svd
//   -ksp_type minres -pc_type none -ksp_converged_reason
// and fieldsplit below

// Jacobian is correct and symmetric:
// ./well -snes_monitor -ksp_type preonly -pc_type svd -da_refine 3 -snes_type test
// ./well -snes_monitor -ksp_type preonly -pc_type svd -da_refine 3 -mat_is_symmetric

// generate matrix in matlab
// ./well -snes_converged_reason -ksp_type preonly -pc_type svd -da_refine 1 -mat_view ascii:foo.m:ascii_matlab
// then:
// >> M = [whos to get name]
// >> A = M(1:2:end,1:2:end);  BT = M(1:2:end,2:2:end);  B = M(2:2:end,1:2:end);  C = M(2:2:end,2:2:end);  T = [A BT; B C]

/* VICTORY:
./well -snes_converged_reason -snes_monitor -snes_fd_color -da_refine 7 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type SCHUR -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none -ksp_converged_reason
   * see snes example ex70.c; note enum option is "SCHUR" not "schur"
   * note these -ksp_type also work:  gmres, cgs, richardson
   *  ... and converge in 2 iterations (for reasons in: Murphy, Golub, Wathen 2000)
   * converges with -ksp_type minres but in ~50 iterations
*/

// view diagonal blocks with fieldsplit:
// ./well -snes_converged_reason -snes_monitor -ksp_type gmres -da_refine 1 -pc_type fieldsplit -pc_fieldsplit_type SCHUR -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none -fieldsplit_0_ksp_view_mat

/* 40 second run (linux-c-opt) on 8 million grid points:
timer ./well -snes_converged_reason -snes_monitor -da_refine 22 -ksp_type gmres -pc_type fieldsplit -pc_fieldsplit_type SCHUR -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none -ksp_converged_reason -fieldsplit_0_ksp_type cg -fieldsplit_0_pc_type icc
   * also works with -fieldsplit_0_ksp_type gmres
   * also in parallel (no speedup) if -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type icc
*/

#include <petsc.h>

typedef struct {
  double u, p;
} Field;

typedef struct {
    double    H,
              rho,
              g,
              nu;
} AppCtx;

PetscErrorCode ExactSolution(DMDALocalInfo *info, Vec X, AppCtx *user) {
    PetscErrorCode ierr;
    double h, x;
    Field  *aX;
    h  = user->H / (info->mx-1);
    ierr = DMDAVecGetArray(info->da,X,&aX); CHKERRQ(ierr);
    for (int i=info->xs; i<info->xs+info->xm; i++) {
        aX[i].u = 0.0;
        x = h * (i + 0.5);
        if (i < info->mx - 1)
            aX[i].p = user->rho * user->g * (user->H - x);
        else
            aX[i].p = 0.0;
    }
    ierr = DMDAVecRestoreArray(info->da,X,&aX); CHKERRQ(ierr);
    return 0;
}

// the staggered version of residual evaluation F(X)
// grid has p at staggered locations "O" where incompressibility u_x=0 is enforced
//    x_i-1        O        x_i        O        x_i+1        O
//    u_i-1                 u_i                 u_i+1
//               p_i-1                p_i                  p_i+1
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field *X,
                                 Field *F, AppCtx *user) {
    const double h  = user->H / (info->mx-1),
                 h2 = h * h;
    for (int i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) { // bottom of water
            F[i].u = X[i].u;                                              // u(0) = 0
            F[i].p = - (X[i+1].u - 0.0) * h;                              // -u_x(0+1/2) = 0
        } else if (i == 1) {
            F[i].u = - user->nu * (X[i+1].u - 2 * X[i].u + 0.0)           // -nu u_xx(x1) + p_x(x1) = - rho g
                       + (X[i].p - X[i-1].p) * h + user->rho * user->g * h2;
            F[i].p = - (X[i+1].u - X[i].u) * h;                           // - u_x(x1+1/2) = 0
        } else if (i > 1 && i < info->mx - 1) {
            F[i].u = - user->nu * (X[i+1].u - 2 * X[i].u + X[i-1].u)      // -nu u_xx(xi) + p_x(xi) = - rho g
                       + (X[i].p - X[i-1].p) * h + user->rho * user->g * h2;
            F[i].p = - (X[i+1].u - X[i].u) * h;                           // - u_x(xi+1/2) = 0
        } else if (i == info->mx - 1) { // top of water
            F[i].u = - user->nu * (- 2 * X[i].u + 2 * X[i-1].u)           // -nu u_xx(xm-1) + p_x(xm-1) = - rho g
                       + (- 2 * X[i-1].p) * h + user->rho * user->g * h2; // and  u_x(xm-1) = 0  and  p(xm-1) = 0
            F[i].u /= 2;                                                  // for symmetry
            F[i].p = X[i].p;                                              // no actual d.o.f. here
        } else {
            SETERRQ(PETSC_COMM_WORLD,1,"no way to get here");
        }
    }
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, double *X,
                                 Mat J, Mat P, AppCtx *user) {
    PetscErrorCode ierr;
    MatStencil   col[5],row;
    double       v[5];
    const double h  = user->H / (info->mx-1);
    for (int i=info->xs; i<info->xs+info->xm; i++) {
        row.i = i;
        if (i == 0) {
            row.c = 0;  col[0].i = i;    col[0].c = 0;  v[0] = 1.0;
            ierr = MatSetValuesStencil(P,1,&row,1,col,v,INSERT_VALUES); CHKERRQ(ierr);
            row.c = 1;  col[0].i = i+1;  col[0].c = 0;  v[0] = - h;
            ierr = MatSetValuesStencil(P,1,&row,1,col,v,INSERT_VALUES); CHKERRQ(ierr);
        } else if (i == 1) {
            row.c = 0;
            col[0].c = 0;  col[0].i = i;    v[0] = 2.0 * user->nu;
            col[1].c = 0;  col[1].i = i+1;  v[1] = - user->nu;
            col[2].c = 1;  col[2].i = i;    v[2] = h;
            col[3].c = 1;  col[3].i = i-1;  v[3] = - h;
            ierr = MatSetValuesStencil(P,1,&row,4,col,v,INSERT_VALUES); CHKERRQ(ierr);
            row.c = 1;
            col[0].c = 0;  col[0].i = i;    v[0] = h;
            col[1].c = 0;  col[1].i = i+1;  v[1] = - h;
            ierr = MatSetValuesStencil(P,1,&row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
        } else if (i > 1 && i < info->mx - 1) {
            row.c = 0;
            col[0].c = 0;  col[0].i = i;    v[0] = 2.0 * user->nu;
            col[1].c = 0;  col[1].i = i-1;  v[1] = - user->nu;
            col[2].c = 0;  col[2].i = i+1;  v[2] = - user->nu;
            col[3].c = 1;  col[3].i = i;    v[3] = h;
            col[4].c = 1;  col[4].i = i-1;  v[4] = - h;
            ierr = MatSetValuesStencil(P,1,&row,5,col,v,INSERT_VALUES); CHKERRQ(ierr);
            row.c = 1;
            col[0].c = 0;  col[0].i = i;    v[0] = h;
            col[1].c = 0;  col[1].i = i+1;  v[1] = - h;
            ierr = MatSetValuesStencil(P,1,&row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
        } else if (i == info->mx - 1) {
            row.c = 0;
            col[0].c = 0;  col[0].i = i;    v[0] = user->nu;
            col[1].c = 0;  col[1].i = i-1;  v[1] = - user->nu;
            col[2].c = 1;  col[2].i = i-1;  v[2] = - h;
            ierr = MatSetValuesStencil(P,1,&row,3,col,v,INSERT_VALUES); CHKERRQ(ierr);
            row.c = 1;
            col[0].c = 1;  col[0].i = i;    v[0] = 1.0;
            ierr = MatSetValuesStencil(P,1,&row,1,col,v,INSERT_VALUES); CHKERRQ(ierr);
        } else {
            SETERRQ(PETSC_COMM_WORLD,1,"no way to get here");
        }
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}


int main(int argc,char **args) {
    PetscErrorCode ierr;
    DM            da;
    SNES          snes;
    AppCtx        user;
    Vec           X, Xexact;
    double        uerrnorm, perrnorm, pnorm;
    DMDALocalInfo info;

    PetscInitialize(&argc,&args,NULL,help);
    user.rho = 1000.0;
    user.g   = 9.81;
    user.H   = 10.0;
    user.nu  = 1.0;   // Pa s;  1.0 for corn syrup; 10^-3 for liquid water

    //ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"wel_","options for well",""); CHKERRQ(ierr);
    //ierr = PetscOptionsBool("-xxx","xxx","well.c",X,&(X),NULL); CHKERRQ(ierr);
    //ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,2,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&X); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = VecSet(X,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,X); CHKERRQ(ierr);

    ierr = VecDuplicate(X,&Xexact); CHKERRQ(ierr);
    ierr = ExactSolution(&info,Xexact,&user); CHKERRQ(ierr);

//ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//ierr = VecView(Xexact,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = VecAXPY(X,-1.0,Xexact); CHKERRQ(ierr);    // X <- X + (-1.0) Xexact
    ierr = VecStrideNorm(X,0,NORM_INFINITY,&uerrnorm); CHKERRQ(ierr);
    ierr = VecStrideNorm(X,1,NORM_INFINITY,&perrnorm); CHKERRQ(ierr);
    ierr = VecStrideNorm(Xexact,1,NORM_INFINITY,&pnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "on %d point grid with h=%g:\n"
           "  |u-uexact|_inf = %.3e,  |p-pexact|_inf / |pexact|_inf = %.3e\n",
           info.mx,user.H/(info.mx-1),uerrnorm,perrnorm/pnorm); CHKERRQ(ierr);

    VecDestroy(&X);  VecDestroy(&Xexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

