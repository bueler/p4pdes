static char help[] =
"Solves a 2D structured-grid advection-diffusion problem with DMDA\n"
"and SNES.  The problem is Example 3.1.4 in Elman et al (2005),\n"
"a 'double-glazing' problem.  The equation is\n"
"    - eps Laplacian u + W . Grad u = 0\n"
"on the domain  [-1,1]^2 where  W(x,y)  corresponds to a recirculating\n"
"flow.  Boundary conditions are:\n"
"    u(1,y) = 1\n"
"    u(-1,y) = u(x,-1) = u(x,1) = 0\n\n";

/* reproduce Figure 3.5 from Elman et al (2005); run takes a couple of minutes
(and preconditioner choice needs more care):
$ timer ./glaze -da_refine 7 -glaze_eps 0.005 -ksp_rtol 1.0e-14 -snes_monitor -snes_converged_reason -ksp_converged_reason -ksp_gmres_restart 400 -snes_monitor_solution ascii:glaze.m:ascii_matlab
$ octave    # or matlab
>> glaze
>> x = linspace(-1,1,257);  u = reshape(u_vec,257,257)';
>> mesh(x,x,u,'edgecolor','k'),  xlabel x,  ylabel y
*/

#include <petsc.h>

typedef struct {
    DM        da;
    PetscReal eps;
} Ctx;

PetscErrorCode configureCtx(Ctx *usr) {
    PetscErrorCode  ierr;
    usr->eps = 1.0;
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


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **F, Ctx *usr) {
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
                F[j][i] = u[j][i] - 1.0;
            } else if (i == 0 || j == 0 || j == info->my-1) {
                F[j][i] = u[j][i];
            } else {
                uu = u[j][i];
                uxx = (u[j][i-1] - 2.0 * uu + u[j][i+1]) / s.hx2;
                uyy = (u[j-1][i] - 2.0 * uu + u[j+1][i]) / s.hy2;
                W = getWind(x,y);
                Wux = W.x * (u[j][i+1] - u[j][i-1]) / (2.0 * s.hx);
                Wuy = W.y * (u[j+1][i] - u[j-1][i]) / (2.0 * s.hy);
                F[j][i] = - e * (uxx + uyy) + Wux + Wuy;
            }
        }
    }
    return 0;
}


PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***u,
                                 Mat J, Mat Jpre, Ctx *usr) {
    PetscErrorCode  ierr;
    PetscInt        i,j,q;
    PetscReal       v[5],diag,x,y;
    const PetscReal e = usr->eps;
    MatStencil      col[5],row;
    Spacings        s;
    Wind            W;

    getSpacings(info,&s);
    diag = e * 2.0 * (1.0/s.hx2 + 1.0/s.hy2);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + j * s.hy;
        row.j = j;
        col[0].j = j;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + i * s.hx;
            row.i = i;
            col[0].i = i;
            q = 1;
            if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
                v[0] = 1.0;
            } else {
                W = getWind(x,y);
                v[0] = diag;
                if (i-1 != 0) {
                    v[q] = - e / s.hx2 - W.x / (2.0 * s.hx);
                    col[q].j = j;  col[q].i = i-1;
                    q++;
                }
                if (i+1 != info->mx-1) {
                    v[q] = - e / s.hx2 + W.x / (2.0 * s.hx);
                    col[q].j = j;  col[q].i = i+1;
                    q++;
                }
                if (j-1 != 0) {
                    v[q] = - e / s.hy2 - W.y / (2.0 * s.hy);
                    col[q].j = j-1;  col[q].i = i;
                    q++;
                }
                if (j+1 != info->my-1) {
                    v[q] = - e / s.hy2 + W.y / (2.0 * s.hy);
                    col[q].j = j+1;  col[q].i = i;
                    q++;
                }
            }
            ierr = MatSetValuesStencil(Jpre,1,&row,q,col,v,INSERT_VALUES); CHKERRQ(ierr);
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
    Vec            u;
    DMDALocalInfo  info;
    Ctx            user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = configureCtx(&user); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,
                -3,-3,
                PETSC_DECIDE,PETSC_DECIDE,
                1,1,
                NULL,NULL,
                &user.da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(user.da,-1.0,1.0,-1.0,1.0,0.0,1.0); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
    if ((info.mx < 2) || (info.my < 2)) {
        SETERRQ(PETSC_COMM_WORLD,1,"grid too coarse ... require (mx,my) > (2,2)");
    }

    ierr = DMCreateGlobalVector(user.da,&u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u_vec");CHKERRQ(ierr);
    ierr = formInitial(&info,&user,u); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(user.da,
            (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d x %d grid with eps=%g ...\n",
         info.mx,info.my,info.mz,user.eps); CHKERRQ(ierr);

    VecDestroy(&u);
    SNESDestroy(&snes);  DMDestroy(&user.da);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

