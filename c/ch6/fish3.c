static char help[] =
"Solves a 3D structured-grid diffusion-advection problem with DMDA\n"
"and SNES.  The equation is\n"
"    - eps Laplacian u + W . Grad u = f\n"
"on the domain  [-1,1]^3\n"
"FIXME: for now eps=1 and W=0 and f=0\n"
"The boundary conditions are\n"
"   u(1,y,z) = g(y,z)\n"
"   u(-1,y,z) = u(x,-1,z) = u(x,1,z) = 0\n"
"   u periodic in z\n"
"The exact solution is (FIXME: make optional)\n"
"   u(x,y,z) = C sinh(D (x+1)) sin(E (y+1)) sin(F (z+1))\n"
"and\n"
"   g(y,z) = sin(E (y+1)) sin(F (z+1)).\n\n";

/* evidence for convergence:
$ for LEV in 0 1 2 3 4; do ./fish3 -ksp_rtol 1.0e-14 -snes_monitor -snes_converged_reason -da_refine $LEV; done
*/

/* in this version, these work???:
  ./fish3 -snes_monitor -ksp_monitor
  ./fish3 -snes_monitor -ksp_monitor -pc_type lu
  ./fish3 -snes_monitor -ksp_monitor -ksp_type cg -pc_type icc
  ./fish3 -snes_monitor -ksp_monitor -snes_fd
*/

#include <petsc.h>

typedef struct {
  DM        da;
  Vec       g,f;
} Ctx;

typedef struct {
  PetscReal hx, hy, hz, hx2, hy2, hz2;
} Spacings;

void fillSpacings(DMDALocalInfo *info, Spacings *s) {
    s->hx = 2.0/(info->mx-1);
    s->hy = 2.0/(info->my-1);
    s->hz = 2.0/(info->mz);    // periodic direction
    s->hx2 = s->hx * s->hx;
    s->hy2 = s->hy * s->hy;
    s->hz2 = s->hz * s->hz;
}

PetscErrorCode formExact(DMDALocalInfo *info, Ctx *usr, Vec uex, Vec g) {
    PetscErrorCode  ierr;
    PetscInt        i, j, k;
    Spacings        s;
    const PetscReal D = sqrt(17.0) * PETSC_PI / 2.0,
                    C = 1.0 / sinh(2.0 * D),
                    E = PETSC_PI / 2.0,
                    F = 2.0 * PETSC_PI;
    PetscReal       x, y, z, gg, ***auex, ***ag;

    fillSpacings(info,&s);
    ierr = DMDAVecGetArray(usr->da, uex, &auex);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(usr->da, g, &ag);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * s.hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + j * s.hy;
            gg = sin(E*(y+1.0)) * sin(F*(z+1.0));
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * s.hx;
                auex[k][j][i] = C * sinh(D*(x+1.0)) * gg;
                if (i == info->mx-1)
                    ag[k][j][i] = gg;
                else
                    ag[k][j][i] = 0.0;
            }
        }
    }
    ierr = DMDAVecRestoreArray(usr->da, uex, &auex);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(usr->da, g, &ag);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(g); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(uex); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(g); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(uex); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal ***u,
                                 PetscReal ***F, Ctx *usr) {
    PetscErrorCode  ierr;
    PetscInt        i, j, k;
    PetscReal       uxx, uyy, uzz, ***af, ***ag;
    Spacings        s;

    fillSpacings(info,&s);
    ierr = DMDAVecGetArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(usr->da, usr->g, &ag);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        for (j=info->ys; j<info->ys+info->ym; j++) {
            for (i=info->xs; i<info->xs+info->xm; i++) {
                if (i == info->mx-1) {
                    F[k][j][i] = u[k][j][i] - ag[k][j][i];
                } else if (i == 0 || j == 0 || j == info->my-1) {
                    F[k][j][i] = u[k][j][i];
                } else {
                    uxx = (u[k][j][i-1] - 2.0 * u[k][j][i] + u[k][j][i+1]) / s.hx2;
                    uyy = (u[k][j-1][i] - 2.0 * u[k][j][i] + u[k][j+1][i]) / s.hy2;
                    uzz = (u[k-1][j][i] - 2.0 * u[k][j][i] + u[k+1][j][i]) / s.hz2;
                    F[k][j][i] = - uxx - uyy - uzz - af[k][j][i];
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(usr->da, usr->g, &ag);CHKERRQ(ierr);
    return 0;
}


PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***u,
                                 Mat J, Mat Jpre, Ctx *usr) {
    PetscErrorCode  ierr;
    PetscInt        i,j,k,q;
    PetscReal       v[7],diag;
    MatStencil      col[7],row;
    Spacings        s;

    fillSpacings(info,&s);
    diag = 2.0*(1.0/s.hx2 + 1.0/s.hy2 + 1.0/s.hz2);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        row.k = k;
        col[0].k = k;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            row.j = j;
            col[0].j = j;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                row.i = i;
                col[0].i = i;
                if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
                    v[0] = 1.0;
                    q = 1;
                } else {
                    v[0] = diag;
                    col[1].k = k-1;  col[1].j = j;  col[1].i = i;
                    v[1] = - 1.0/s.hz2;
                    col[2].k = k+1;  col[2].j = j;  col[2].i = i;
                    v[2] = - 1.0/s.hz2;
                    q = 3;
                    if (i-1 != 0) {
                        v[q] = - 1.0/s.hx2;
                        col[q].k = k;  col[q].j = j;  col[q].i = i-1;
                        q++;
                    }
                    if (i+1 != info->mx-1) {
                        v[q] = - 1.0/s.hx2;
                        col[q].k = k;  col[q].j = j;  col[q].i = i+1;
                        q++;
                    }
                    if (j-1 != 0) {
                        v[q] = - 1.0/s.hy2;
                        col[q].k = k;  col[q].j = j-1;  col[q].i = i;
                        q++;
                    }
                    if (j+1 != info->my-1) {
                        v[q] = - 1.0/s.hy2;
                        col[q].k = k;  col[q].j = j+1;  col[q].i = i;
                        q++;
                    }
                }
                ierr = MatSetValuesStencil(Jpre,1,&row,q,col,v,INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    return 0;
}


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    SNES           snes;
    Vec            u, uexact;
    PetscReal      errnorm, uexnorm;
    DMDALocalInfo  info;
    Ctx            user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC,
                DMDA_STENCIL_STAR,
                -3,-3,-3,
                PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                1,1,
                NULL,NULL,NULL,
                &user.da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(user.da,-1.0,1.0,-1.0,1.0,-1.0,1.0); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
    if ((info.mx < 2) || (info.my < 2) || (info.mz < 3)) {
        SETERRQ(PETSC_COMM_WORLD,1,"grid too coarse ... require (mx,my,mz) > (2,2,3)");
    }

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(user.da,
            (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(user.da,&u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&user.f); CHKERRQ(ierr);
    ierr = VecSet(user.f,0.0); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    ierr = VecDuplicate(u,&user.g); CHKERRQ(ierr);
    ierr = formExact(&info,&user,uexact,user.g); CHKERRQ(ierr);
    ierr = VecCopy(user.g,u); CHKERRQ(ierr);   // g has zeros except at bdry

    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
    ierr = VecNorm(u,NORM_2,&errnorm); CHKERRQ(ierr);
    ierr = VecNorm(uexact,NORM_2,&uexnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d x %d grid:  error |u-uexact|_2/|uexact|_2 = %g\n",
             info.mx,info.my,info.mz,errnorm/uexnorm); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&user.f);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&user.da);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

