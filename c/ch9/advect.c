static char help[] =
"Solves time-dependent pure-advection equation in 2D using TS.  Option prefix -adv_.\n"
"Equation is  u_t + div(a(x,y) u) = f.  Domain is (0,1) x (0,1).\n"
"Boundary conditions are periodic in x and y and cells are grid-point centered.\n"
"Allows comparison of first-order upwind and flux-limited (non-oscillatory)\n"
"finite difference method-of-lines discretizations.\n";

#include <petsc.h>

typedef struct {
    PetscBool  circlewind;  // if true, wind is equivalent to rigid rotation
    double     angle;       // angle of wind relative to right-ward, in degrees
} AdvectCtx;

/* method-of-lines discretization gives ODE system  u' = G(t,u) */
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, double **au,
                                    double **aG, AdvectCtx *user) {
    int      i, j;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            aG[j][i] = FIXME;
        }
    }
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

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "adv_", "options for advect.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-angle","angle of wind relative to right-ward, in degrees",
           "advect.c",user.angle,&user.angle,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-circlewind","if true, wind is equivalent to rigid rotation",
           "advect.c",user.circlewind,&user.circlewind,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
               4,4,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.25 grid
               1,1,                       // degrees of freedom, stencil width
               NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 1.0 / info.mx;  hy = 1.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,0.0+hx/2.0,1.0-hx/2.0,
                                        0.0+hy/2.0,1.0-hy/2.0,0.0,1.0);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK2A); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(ts,0.0,0.1); CHKERRQ(ierr);
    ierr = TSSetDuration(ts,1000000,1.0); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "solving on %d x %d grid with dx=%g x dy=%g cells, t0=%g,\n"
           "and initial step dt=%g ...\n",
           info.mx,info.my,hx,hy,t0,dt); CHKERRQ(ierr);

    ierr = VecSet(u,0.0); CHKERRQ(ierr);   // initial condition  FIXME
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

