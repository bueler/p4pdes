static char help[] = "Solves a 3D structured-grid Poisson problem with DMDA\n"
"and SNES.\n\n";

/* in this version, these work:
  ./fish3 -f3_exactinit -snes_fd
  ./fish3 -f3_exactinit -snes_fd
  ./fish3  // only works because Jacobian never used
but not:
  ./fish3 -f3_exactinit
*/

#include <petsc.h>

PetscErrorCode formExact(DM da, Vec uexact) {
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscInt       i, j, k;
    PetscReal      hx, hy, hz, x, y, z, ***auexact;

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);  hz = 1.0/(info.mz-1);
    ierr = DMDAVecGetArray(da, uexact, &auexact);CHKERRQ(ierr);
    for (k=info.zs; k<info.zs+info.zm; k++) {
        z = k * hz;
        for (j=info.ys; j<info.ys+info.ym; j++) {
            y = j * hy;
            for (i=info.xs; i<info.xs+info.xm; i++) {
                x = i * hx;
                auexact[k][j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0) * z*z * (z*z - 1.0);
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, uexact, &auexact);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal ***u,
                                 PetscReal ***F, void *user) {
    PetscInt       i, j, k;
    PetscReal      hx, hy, hz, x, y, z, uxx, uyy, uzz, vol, rhs;

    hx = 1.0/(info->mx-1);  hy = 1.0/(info->my-1);  hz = 1.0/(info->my-1);
    vol = hx * hy * hz;
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = k * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = j * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = i * hx;
                if (i == 0 || j == 0 || k == 0
                    || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
                    F[k][j][i] = u[k][j][i];
                } else {
                    uxx = (u[k][j][i-1] - 2.0 * u[k][j][i] + u[k][j][i+1]) / (hx*hx);
                    uyy = (u[k][j-1][i] - 2.0 * u[k][j][i] + u[k][j+1][i]) / (hy*hy);
                    uzz = (u[k-1][j][i] - 2.0 * u[k][j][i] + u[k+1][j][i]) / (hy*hy);
                    rhs = 0.0 * x * y * z;// FIXME
                    F[k][j][i] = vol * (rhs - uxx - uyy - uzz);
                    // EXERCISE: make more efficient by NOT doing "divide by hx^2 then mult by vol"
                    //           and instead "mult by hy*hz/hx"; measure performance difference
                }
            }
        }
    }
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***u,
                                 Mat J, Mat Jpre, void *user) {
    SETERRQ(PETSC_COMM_WORLD,1,"NOT IMPLEMENTED");
    return 0;
}

//MAIN
int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    PetscBool      exactinit = PETSC_FALSE;
    PetscReal      errnorm;
    DMDALocalInfo  info;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"f3_","options to fish3","");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exactinit","use exact solution as initial iterate",
                    NULL,exactinit,&exactinit,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,
                -5,-5,-5,
                PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                1,1,
                NULL,NULL,NULL,
                &da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,
            NULL);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
            (PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,
            NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    if (exactinit) {
      ierr = formExact(da,u); CHKERRQ(ierr);
    } else {
      ierr = VecSet(u,0.0); CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    ierr = formExact(da,uexact); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d x %d grid:  error |u-uexact|_inf = %g\n",
             info.mx,info.my,info.mz,errnorm); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}
//ENDMAIN

