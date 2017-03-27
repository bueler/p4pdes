static char help[] =
"Solves time-dependent pure-advection equation in 2D using TS.  Option prefix -adv_.\n"
"Equation is  u_t + div(a(x,y) u) = f.  Domain is (0,1) x (0,1).\n"
"Boundary conditions are periodic in x and y.\n"
"Allows comparison of first-order upwind and flux-limited (non-oscillatory)\n"
"finite difference method-of-lines discretizations.\n";

#include <petsc.h>

typedef struct {
    PetscBool  circlewind;
    double     windangle;
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
    Vec            u;
    DMDALocalInfo  info;
    double         hx, hy, hxhy, t0, dt;

    PetscInitialize(&argc,&argv,(char*)0,help);

    FIXME
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "heat_", "options for heat", ""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-D0","constant thermal diffusivity",
           "heat.c",user.D0,&user.D0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor","also display total heat energy at each step",
           "heat.c",monitorenergy,&monitorenergy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
               5,4,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.25 grid
               1,1,                       // degrees of freedom, stencil width
               NULL,NULL,&user.da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
    ierr = DMSetUp(user.da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(user.da,&u); CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSFIXME); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(ts,0.0,0.01); CHKERRQ(ierr);
    ierr = TSSetDuration(ts,1000000,0.1); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
    ierr = Spacings(user.da,&hx,&hy); CHKERRQ(ierr);
    hxhy = PetscMin(hx,hy);  hxhy = hxhy * hxhy;
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "solving on %d x %d grid with dx=%g x dy=%g cells, t0=%g,\n"
           "and initial step dt=%g ...\n",
           info.mx,info.my,hx,hy,t0,dt); CHKERRQ(ierr);

    ierr = VecSet(u,0.0); CHKERRQ(ierr);   // initial condition
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&user.da);
    PetscFinalize();
    return 0;
}

