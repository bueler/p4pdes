static char help[] =
"Solves time-dependent pure-advection equation in 2D using TS.  Option prefix -adv_.\n"
"Equation is  u_t + div(a(x,y) u) = f.  Domain is (0,1) x (0,1).\n"
"Boundary conditions are periodic in x and y and cells are grid-point centered.\n"
"Allows comparison of first-order upwind and flux-limited (non-oscillatory)\n"
"finite difference method-of-lines discretizations.\n";

#include <petsc.h>

// try:
//   ./advect -da_refine 5 -ts_monitor_solution draw -ts_monitor -ts_rk_type 5dp

// reproduce Figure 2.3 (left) in Hundsdorfer & Verwer:  FIXME: only first-order upwind for now
//   export ADVOPTS="-da_grid_x 100 -da_grid_y 100 -ts_monitor_solution draw -ts_monitor -ts_adapt_type none -ts_dt 0.003"
//   ./advect $ADVOPTS
//   ./advect $ADVOPTS -adv_windx 1.0 -adv_windy 0.0

// exercise: evaluate error by reusing FormInitial()

typedef struct {
    PetscBool  circlewind;   // if true, wind is equivalent to rigid rotation
    double     windx, windy, // x,y components of wind (if not circular)
               conex, coney, coner, coneh; // parameters for cone initial cond.
} AdvectCtx;

PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, AdvectCtx* user) {
    PetscErrorCode ierr;
    DMDACoor2d   **coords;
    int          i, j;
    double       x, y, r, **au;

    ierr = VecSet(u,0.0); CHKERRQ(ierr);  // clear it first
    ierr = DMDAGetCoordinateArray(info->da, &coords); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = coords[j][i].x - user->conex;
            y = coords[j][i].y - user->coney;
            r = PetscSqrtReal(x * x + y * y);
            if (r < user->coner) {
                au[j][i] = user->coneh * (1.0 - r / user->coner);
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
    ierr = DMDARestoreCoordinateArray(info->da, &coords); CHKERRQ(ierr);
    return 0;
}

static double a_wind(double x, double y, int dir, AdvectCtx* user) {
    if (user->circlewind) {
        return (dir == 0) ? -y : x;
    } else {
        return (dir == 0) ? user->windx : user->windy;
    }
}

static double g_source(double x, double y, double u, AdvectCtx* user) {
    return 0.0;
}

/* method-of-lines discretization gives ODE system  u' = G(t,u)
FIXME only first-order upwind for now */
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, double **au,
                                    double **aG, AdvectCtx *user) {
    PetscErrorCode ierr;
    DMDACoor2d   **coords;
    int          i, j, l;
    const int    dir[4] = {0, 1, 0, 1},
                 upi[4] = {0, 0, 1, 0},  dni[4] = {1, 0, 0, 0},
                 upj[4] = {0, 0, 0, 1},  dnj[4] = {0, 1, 0, 0};
    double       hx, hy, x, y, a,
                 xshift[4], yshift[4], flux[4];

    // to compute fluxes on boundaries we traverse midpoints of cell boundaries
    // in this order:
    //    ---1---
    //   |       |
    //   2   *   0
    //   |       |
    //    ---3---
    hx = 1.0 / info->mx;  hy = 1.0 / info->my;
    xshift[0] =   hx / 2.0;  yshift[0] = 0.0;
    xshift[1] =        0.0;  yshift[1] =   hy / 2.0;
    xshift[2] = - hx / 2.0;  yshift[2] = 0.0;
    xshift[3] =        0.0;  yshift[3] = - hy / 2.0;
    ierr = DMDAGetCoordinateArray(info->da, &coords); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = coords[j][i].x;  y = coords[j][i].y;
            for (l = 0; l < 4; l++) {
                a = a_wind(x + xshift[l],y + yshift[l],dir[l],user);
                flux[l] = a * ( (a >= 0) ? au[j-upj[l]][i-upi[l]]
                                         : au[j+dnj[l]][i+dni[l]] );
            }
            aG[j][i] = - (flux[0] - flux[2]) / hx - (flux[1] - flux[3]) / hy
                       + g_source(x,y,au[j][i],user);
        }
    }
    ierr = DMDARestoreCoordinateArray(info->da, &coords); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    AdvectCtx      user;
    TS             ts;
    DM             da;
    Vec            u;
    DMDALocalInfo  info;
    double         hx, hy, t0, dt;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.circlewind = PETSC_FALSE;
    user.windx = 1.0;
    user.windy = 1.0;

    // the cone initial condition is for reproducing results from
    // Hundsdorfer & Verwer, "Numerical Solution of Time-Dependent Advection-
    // Diffusion-Reaction Equations", Springer 2003, page 303
    // ... these could be set from options
    user.conex = 0.2;
    user.coney = 0.2;
    user.coner = 0.1;
    user.coneh = 1.0;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "adv_", "options for advect.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-circlewind","if true, wind is equivalent to rigid rotation",
           "advect.c",user.circlewind,&user.circlewind,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windx","x component of wind (if not circular)",
           "advect.c",user.windx,&user.windx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windy","y component of wind (if not circular)",
           "advect.c",user.windy,&user.windy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
               DMDA_STENCIL_STAR,              // no diagonal differencing
               4,4,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.25 grid
               1,                              // degrees of freedom
               2,                              // stencil width needed for flux-limiting
               NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 1.0 / info.mx;  hy = 1.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,0.0+hx/2.0,1.0-hx/2.0,
                                        0.0+hy/2.0,1.0-hy/2.0,0.0,1.0);CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts,TSRK2A); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(ts,0.0,0.1); CHKERRQ(ierr);
    ierr = TSSetDuration(ts,1000000,0.6); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "solving on %d x %d grid with dx=%g x dy=%g cells, t0=%g,\n"
           "and initial step dt=%g ...\n",
           info.mx,info.my,hx,hy,t0,dt); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = FormInitial(&info,u,&user); CHKERRQ(ierr);
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

