static char help[] =
"1D Stokes problem with DMDA and SNES.  Option prefix -wel_.\n\n";

// staggered might actually be right:
// ./well -snes_monitor -snes_converged_reason -ksp_type preonly -pc_type svd -snes_fd_color -da_refine 7 -snes_monitor_solution draw -draw_pause 2

// try:
//   -ksp_type preonly -pc_type svd
//   -ksp_type minres -pc_type none -ksp_converged_reason
// and fieldsplit below

// matrix is really symmetric:
// ./well -snes_monitor -snes_fd_color -ksp_type preonly -pc_type svd -da_refine 3 -mat_is_symmetric 1.0e-15

// view diagonal blocks with fieldsplit:
// ./well -snes_converged_reason -snes_monitor -snes_fd_color -ksp_type minres -da_refine 1 -pc_type fieldsplit -fieldsplit_0_ksp_view_mat

// generate matrix in matlab
// ./well -snes_monitor -snes_fd -snes_converged_reason -ksp_type preonly -pc_type svd -da_refine 1 -mat_view ascii:foo.m:ascii_matlab
// then:
// >> M = [whos to get name]
// >> A = M(1:2:end,1:2:end);  B = M(1:2:end,2:2:end);  BT = M(2:2:end,1:2:end);  C = M(2:2:end,2:2:end);
// >> T = [A B; BT C]

/* VICTORY:
./well -snes_converged_reason -snes_monitor -snes_fd_color -da_refine 7 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type SCHUR -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none -ksp_converged_reason
   * see snes example ex70.c; note enum option is "SCHUR" not "schur"
   * note these -ksp_type also work:  gmres, cgs, richardson
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
    h  = user->H / (info->mx-2);
    ierr = DMDAVecGetArray(info->da,X,&aX); CHKERRQ(ierr);
    for (int i=info->xs; i<info->xs+info->xm; i++) {
        aX[i].u = 0.0;
        x = h * (i + 0.5);
        aX[i].p = user->rho * user->g * (user->H - x);
    }
    ierr = DMDAVecRestoreArray(info->da,X,&aX); CHKERRQ(ierr);
    return 0;
}

// the staggered version:
//    x_i-1        O        x_i        O        x_i+1
//    u_i-1      p_i-1      u_i       p_i       u_i+1
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field *X,
                                 Field *F, AppCtx *user) {
    const double h     = user->H / (info->mx-2),
                 h2    = h * h;
    for (int i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            F[i].u = X[i].u;                                              // u(0) = 0
            F[i].p = - (X[i+1].u - 0.0) / h;                              // -u_x(0+1/2) = 0
        } else if (i == 1) {
            F[i].u = -user->nu * (X[i+1].u - 2 * X[i].u + 0.0) / h2       // -nu u_xx(x1) + p_x(x1) = - rho g
                       + (X[i].p - X[i-1].p) / h + user->rho * user->g;
            F[i].p = - (X[i+1].u - X[i].u) / h;                           // - u_x(x1+1/2) = 0
        } else if (i > 1 && i < info->mx - 1) {
            F[i].u = -user->nu * (X[i+1].u - 2 * X[i].u + X[i-1].u) / h2  // -nu u_xx(xi) + p_x(xi) = - rho g
                       + (X[i].p - X[i-1].p) / h + user->rho * user->g;
            F[i].p = - (X[i+1].u - X[i].u) / h;                           // - u_x(xi+1/2) = 0
        } else if (i == info->mx - 1) {
            F[i].u = -user->nu * (- X[i].u + X[i-1].u) / h2               // -nu/2 u_xx(xm-1) + p_x(xm-1)/2 = - rho g/2
                       + (- X[i-1].p) / h + user->rho * user->g / 2;      // and  u_x(xm-1) = 0
            F[i].p = X[i].p;                                              // p(xm-1/2) = 0
        } else {
            SETERRQ(PETSC_COMM_WORLD,1,"no way to get here");
        }
    }
    return 0;
}

/*  from c/ch4/reaction.c
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, double *u,
                                 Mat J, Mat P, AppCtx *user) {
    PetscErrorCode ierr;
    int    i, col[3];
    double h = 1.0 / (info->mx-1), dRdu, v[3];
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            ierr = MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES); CHKERRQ(ierr);
        } else {
            dRdu = - (user->rho / 2.0) / PetscSqrtReal(u[i]);
            col[0] = i-1;  v[0] = - 1.0;
            col[1] = i;
            if (user->noRinJ)
                v[1] = 2.0;
            else
                v[1] = 2.0 - h*h * dRdu;
            col[2] = i+1;  v[2] = - 1.0;
            ierr = MatSetValues(P,1,&i,3,col,v,INSERT_VALUES); CHKERRQ(ierr);
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
*/

int main(int argc,char **args) {
    PetscErrorCode ierr;
    DM            da;
    SNES          snes;
    AppCtx        user;
    Vec           u, uexact;
    double        errnorm, unorm;
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

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    //ierr = DMDASNESSetJacobianLocal(da,
    //           (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    ierr = ExactSolution(&info, uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
    ierr = VecNorm(uexact,NORM_INFINITY,&unorm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "on %d point grid:  |u-uexact|_inf / |uexact|_inf = %g\n",
           info.mx,errnorm/unorm); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

