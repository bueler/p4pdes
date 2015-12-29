static char help[] =
"Solves a 2D structured-grid advection-diffusion problem with DMDA\n"
"and SNES.  The problem is Example 3.1.4 in Elman et al (2005),\n"
"a double-glazing problem.\n\n";

#include <petsc.h>

typedef struct {
    DM        da;
    PetscReal eps;
} Ctx;

PetscErrorCode configureCtx(Ctx *usr) {
    PetscErrorCode  ierr;
    usr->eps = 1.0;
    usr->upwind = PETSC_FALSE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"glaze_","2D advection-diffusion solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","diffusion coefficient eps with  0 < eps < infty",
               NULL,usr->eps,&(usr->eps),NULL); CHKERRQ(ierr);
    if (usr->eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",usr->eps);
    }
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}


typedef struct {
    PetscReal x,y;
} Wind;

Wind getWind(PetscReal x, PetscReal y) {
    Wind W = {2.0*y*(1.0-x*x), -2.0*x*(1.0-y*y)};
    return W;
}


typedef struct {
    PetscReal hx, hy, hx2, hy2;
} Spacings;

void getSpacings(DMDALocalInfo *info, Spacings *s) {
    s->hx = 2.0/(info->mx-1);
    s->hy = 2.0/(info->my-1);
    s->hx2 = s->hx * s->hx;
    s->hy2 = s->hy * s->hy;
}


PetscErrorCode formInitial(DMDALocalInfo *info, Ctx *usr, Vec u0) {
    PetscErrorCode  ierr;
    PetscInt        i, j;
    PetscReal       **au0;

    ierr = DMDAVecGetArray(usr->da, u0, &au0);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        for (i=info->xs; i<info->xs+info->xm; i++) {
            if (i == info->mx-1)
                au0[j][i] = 1.0;
            else
                au0[j][i] = 0.0;
        }
    }
    ierr = DMDAVecRestoreArray(usr->da, u0, &au0);CHKERRQ(ierr);
    return 0;
}


FIXME from here

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **F, Ctx *usr) {
    PetscErrorCode  ierr;
    PetscInt        i, j;
    const PetscReal e = usr->eps;
    PetscReal       x, y, uu, uxx, uyy, Wux, Wuy;
    Wind            W;
    Spacings        s;

    getSpacings(info,&s);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + j * s.hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + i * s.hx;
            if (i == info->mx-1) {
                F[k][j][i] = u[k][j][i] - ag[k][j][i];
            } else if (i == 0 || j == 0 || j == info->my-1) {
                F[k][j][i] = u[k][j][i];
            } else {
                uu = u[k][j][i];
                uxx = (u[k][j][i-1] - 2.0 * uu + u[k][j][i+1]) / s.hx2;
                uyy = (u[k][j-1][i] - 2.0 * uu + u[k][j+1][i]) / s.hy2;
                uzz = (u[k-1][j][i] - 2.0 * uu + u[k+1][j][i]) / s.hz2;
                W = getWind(x,y,z);
                Wux = W.x * (u[k][j][i+1] - u[k][j][i-1]) / (2.0 * s.hx);
                Wuy = W.y * (u[k][j+1][i] - u[k][j-1][i]) / (2.0 * s.hy);
                Wuz = W.z * (u[k+1][j][i] - u[k-1][j][i]) / (2.0 * s.hz);
                F[k][j][i] = - e * (uxx + uyy + uzz) + Wux + Wuy + Wuz
                             - af[k][j][i];
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
    PetscReal       v[7],diag,x,y,z;
    const PetscReal e = usr->eps;
    MatStencil      col[7],row;
    Spacings        s;
    Wind            W;

    getSpacings(info,&s);
    diag = e * 2.0 * (1.0/s.hx2 + 1.0/s.hy2 + 1.0/s.hz2);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * s.hz;
        row.k = k;
        col[0].k = k;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + j * s.hy;
            row.j = j;
            col[0].j = j;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * s.hx;
                row.i = i;
                col[0].i = i;
                if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
                    v[0] = 1.0;
                    q = 1;
                } else {
                    W = getWind(x,y,z);
                    v[0] = diag;
                    if (usr->upwind) {
                        v[0] += (W.x / s.hx) * ((W.x > 0.0) ? 1.0 : -1.0);
                        v[0] += (W.y / s.hy) * ((W.y > 0.0) ? 1.0 : -1.0);
                        v[0] += (W.z / s.hz) * ((W.z > 0.0) ? 1.0 : -1.0);
                    }
                    v[1] = - e / s.hz2;
                    if (usr->upwind) {
                        if (W.z > 0.0)
                            v[q] -= W.z / s.hz;
                    } else
                        v[q] -= W.z / (2.0 * s.hz);
                    col[1].k = k-1;  col[1].j = j;  col[1].i = i;
                    v[2] = - e / s.hz2;
                    if (usr->upwind) {
                        if (W.z <= 0.0)
                            v[q] += W.z / s.hz;
                    } else
                        v[q] += W.z / (2.0 * s.hz);
                    col[2].k = k+1;  col[2].j = j;  col[2].i = i;
                    q = 3;
                    if (i-1 != 0) {
                        v[q] = - e / s.hx2;
                        if (usr->upwind) {
                            if (W.x > 0.0)
                                v[q] -= W.x / s.hx;
                        } else
                            v[q] -= W.x / (2.0 * s.hx);
                        col[q].k = k;  col[q].j = j;  col[q].i = i-1;
                        q++;
                    }
                    if (i+1 != info->mx-1) {
                        v[q] = - e / s.hx2;
                        if (usr->upwind) {
                            if (W.x <= 0.0)
                                v[q] += W.x / s.hx;
                        } else
                            v[q] += W.x / (2.0 * s.hx);
                        col[q].k = k;  col[q].j = j;  col[q].i = i+1;
                        q++;
                    }
                    if (j-1 != 0) {
                        v[q] = - e / s.hy2;
                        if (usr->upwind) {
                            if (W.y > 0.0)
                                v[q] -= W.y / s.hy;
                        } else
                            v[q] -= W.y / (2.0 * s.hy);
                        col[q].k = k;  col[q].j = j-1;  col[q].i = i;
                        q++;
                    }
                    if (j+1 != info->my-1) {
                        v[q] = - e / s.hy2;
                        if (usr->upwind) {
                            if (W.y <= 0.0)
                                v[q] += W.y / s.hy;
                        } else
                            v[q] += W.y / (2.0 * s.hy);
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
    PetscReal      err, uexnorm;
    DMDALocalInfo  info;
    Ctx            user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = configureCtx(&user); CHKERRQ(ierr);

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

    ierr = DMCreateGlobalVector(user.da,&uexact); CHKERRQ(ierr);
    ierr = VecDuplicate(uexact,&user.f); CHKERRQ(ierr);
    ierr = VecDuplicate(uexact,&user.g); CHKERRQ(ierr);
    ierr = formUexFG(&info,&user,uexact); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(user.da,
            (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = VecDuplicate(uexact,&u); CHKERRQ(ierr);
    ierr = VecCopy(user.g,u); CHKERRQ(ierr);   // g has zeros except at bdry
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
    ierr = VecNorm(u,NORM_2,&err); CHKERRQ(ierr);
    ierr = VecNorm(uexact,NORM_2,&uexnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d x %d grid with eps=%g:  error |u-uexact|_2/|uexact|_2 = %g\n",
         info.mx,info.my,info.mz,user.eps,err/uexnorm); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);
    VecDestroy(&user.f);  VecDestroy(&user.g);
    SNESDestroy(&snes);  DMDestroy(&user.da);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

