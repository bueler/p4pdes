static char help[] =
"Time-dependent pure-advection equation in 2D using TS.  Option prefix -adv_.\n"
"Domain is (0,1) x (0,1).  Equation is\n"
"  u_t + div(a(x,y) u) = g(x,y,u).\n"
"Boundary conditions are periodic in x and y.  Cells are grid-point centered.\n"
"Uses van Leer (1974) flux-limited (non-oscillatory) method-of-lines\n"
"discretization [default], or first-order upwind.\n";

#include <petsc.h>

// try:
//   ./advect -da_refine 5 -ts_monitor_solution draw -ts_monitor -ts_rk_type 5dp

// one lap of circular motion, computed in parallel:
//   mpiexec -n 4 ./advect -da_refine 5 -adv_circlewind -adv_conex 0.3 -adv_coney 0.3 -ts_final_time 3.1416 -ts_monitor -ts_monitor_solution draw

// implicit:
// mpiexec -n 4 ./advect -ts_monitor_solution draw -ts_monitor -adv_circlewind -ts_final_time 0.5 -adv_conex 0.3 -adv_coney 0.3 -ts_type cn -da_refine 6 -snes_monitor -ts_dt 0.05 -ksp_rtol 1.0e-10

// with -adv_firstorder, -snes_type test suggests Jacobian is correct

// testing Jacobian options (fails with XX=-snes_mf_operator; use -ksp_view_mat ::ascii_matlab  etc.):
// ./advect -da_refine 0 -ts_monitor -adv_circlewind -adv_conex 0.3 -adv_coney 0.3 -ts_type beuler -snes_monitor_short -ts_final_time 0.01 -ts_dt 0.01 -snes_rtol 1.0e-4 -adv_firstorder XX

typedef struct {
    PetscBool  firstorder,   // if true, use first-order upwinding
               circlewind;   // if true, wind is equivalent to rigid rotation
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

// velocity  a(x,y) = ( a^x(x,y), a^y(x,y) )
static double a_wind(double x, double y, int dir, AdvectCtx* user) {
    if (user->circlewind) {
        return (dir == 0) ? - 2.0 * (y - 0.5) : 2.0 * (x - 0.5);
    } else {
        return (dir == 0) ? user->windx : user->windy;
    }
}

// source  g(x,y,u)
static double g_source(double x, double y, double u, AdvectCtx* user) {
    return 0.0;
}

//         d g(x,y,u) / d u
static double dg_source(double x, double y, double u, AdvectCtx* user) {
    return 0.0;
}

/* the van Leer (1974) limiter is formula (1.11) in section III.1 of
Hundsdorfer & Verwer */
//STARTLIMITER
static double limiter(double th) {
    return 0.5 * (th + PetscAbsReal(th)) / (1.0 + PetscAbsReal(th));
}
//ENDLIMITER

/* method-of-lines discretization gives ODE system  u' = G(t,u)
so our finite volume scheme computes
    G_ij = - (fluxE - fluxW)/hx - (fluxN - fluxS)/hy + g(x,y,U_ij)
but only east (E) and north (N) fluxes are computed
*/
//STARTFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t,
        double **au, double **aG, AdvectCtx *user) {
    int          i, j, l;
    double       hx, hy, halfx, halfy, x, y, a, u_up, u_dn, u_far, theta, flux;

    // clear G first
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            aG[j][i] = 0.0;
    // fluxes on cell boundaries are traversed in this order (=l): ,-1-,
    // cell center at * has coordinates (x,y):                     | * 0
    //                                                             '---'
    hx = 1.0 / info->mx;  hy = 1.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys-1; j < info->ys + info->ym; j++) {  // note start: -1
        y = (j + 0.5) * hy;
        for (i = info->xs-1; i < info->xs + info->xm; i++) {  // ditto
            x = (i + 0.5) * hx;
            if ((i >= info->xs) && (j >= info->ys)) {
                aG[j][i] += g_source(x,y,au[j][i],user);
            }
            for (l = 0; l < 2; l++) {   // get E, N fluxes on cell boundaries
                if ((l == 0) && (j < info->ys))  continue;
                if ((l == 1) && (i < info->xs))  continue;
                a = a_wind(x + halfx*(1-l),y + halfy*l,l,user);
                // first-order flux
                u_up = (a >= 0.0) ? au[j][i] : au[j+l][i+(1-l)];
                flux = a * u_up;
                // use flux-limiter
                if (!user->firstorder) { // use formulas (1.2), (1.3), (1.6)
                                         // (pp 216--217) Hundsdorfer&Verwer
                    u_dn = (a >= 0.0) ? au[j+l][i+(1-l)] : au[j][i];
                    if (u_dn != u_up) {
                        u_far = (a >= 0.0) ? au[j-l][i-(1-l)]
                                           : au[j+2*l][i+2*(1-l)];
                        theta = (u_up - u_far) / (u_dn - u_up);
                        flux += a * limiter(theta) * (u_dn - u_up);
                    }
                }
                // update G_ij on both sides of computed flux, if we own it
                if (l == 0) {
                    if (i >= info->xs)              aG[j][i]   -= flux / hx;
                    if (i+1 < info->xs + info->xm)  aG[j][i+1] += flux / hx;
                } else {
                    if (j >= info->ys)              aG[j][i]   -= flux / hy;
                    if (j+1 < info->ys + info->ym)  aG[j+1][i] += flux / hy;
                }
            }
        }
    }
    return 0;
}
//ENDFUNCTION

PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo *info, double t,
        double **au, Mat J, Mat P, AdvectCtx *user) {
    PetscErrorCode ierr;
    const int    dir[4] = {0, 1, 0, 1},  // use x (0) or y (1) component
                 xsh[4]   = { 1, 0,-1, 0},  ysh[4]   = { 0, 1, 0,-1};
    int          i, j, l, nc;
    double       hx, hy, halfx, halfy, x, y, a, v[5];
    MatStencil   col[5],row;

    ierr = MatZeroEntries(P); CHKERRQ(ierr);
    hx = 1.0 / info->mx;  hy = 1.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys; j < info->ys+info->ym; j++) {
        y = (j + 0.5) * hy;
        row.j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            x = (i + 0.5) * hx;
            row.i = i;
            col[0].j = j;  col[0].i = i;
            v[0] = dg_source(x,y,au[j][i],user);
            nc = 1;
            for (l = 0; l < 4; l++) {   // loop over cell boundaries
                a = a_wind(x + halfx*xsh[l],y + halfy*ysh[l],dir[l],user);
                switch (l) {
                    case 0:
                        col[nc].j = j;  col[nc].i = (a >= 0.0) ? i : i+1;
                        v[nc++] = - a / hx;
                        break;
                    case 1:
                        col[nc].j = (a >= 0.0) ? j : j+1;  col[nc].i = i;
                        v[nc++] = - a / hy;
                        break;
                    case 2:
                        col[nc].j = j;  col[nc].i = (a >= 0.0) ? i-1 : i;
                        v[nc++] = a / hx;
                        break;
                    case 3:
                        col[nc].j = (a >= 0.0) ? j-1 : j;  col[nc].i = i;
                        v[nc++] = a / hy;
                        break;
                }
            }
            ierr = MatSetValuesStencil(P,1,&row,nc,col,v,ADD_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
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
    PetscBool      dump = PETSC_FALSE;
    char           fileroot[PETSC_MAX_PATH_LEN] = "";

    PetscInitialize(&argc,&argv,(char*)0,help);

    // the wind, and the cone initial condition, are from
    // Hundsdorfer & Verwer, "Numerical Solution of Time-Dependent Advection-
    // Diffusion-Reaction Equations", Springer 2003, page 303
    user.windx = 1.0;
    user.windy = 1.0;
    user.conex = 0.2;
    user.coney = 0.2;
    user.coner = 0.1;
    user.coneh = 1.0;
    user.circlewind = PETSC_FALSE;
    user.firstorder = PETSC_FALSE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,
           "adv_", "options for advect.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-circlewind","if true, wind is rigid rotation",
           "advect.c",user.circlewind,&user.circlewind,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-coneh","cone height",
           "advect.c",user.coneh,&user.coneh,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-coner","cone radius",
           "advect.c",user.coner,&user.coner,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-conex","x component of cone center",
           "advect.c",user.conex,&user.conex,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-coney","y component of cone center",
           "advect.c",user.coney,&user.coney,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-dumpto","filename root for initial/final state",
           "advect.c",fileroot,fileroot,PETSC_MAX_PATH_LEN,&dump);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-firstorder","if true, use first-order upwinding",
           "advect.c",user.firstorder,&user.firstorder,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windx","x component of wind (if not circular)",
           "advect.c",user.windx,&user.windx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windy","y component of wind (if not circular)",
           "advect.c",user.windy,&user.windy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
               DMDA_STENCIL_STAR,              // no diagonal differencing
               5,5,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.2 grid
                                               //   (mx=my=5 allows -snes_fd_color)
               1,                              // degrees of freedom
               2,                              // stencil width (flux-limiting)
               NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    // grid is cell-centered
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 1.0 / info.mx;  hy = 1.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,
        0.0+hx/2.0,1.0-hx/2.0,0.0+hy/2.0,1.0-hy/2.0,0.0,1.0);CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSJacobianLocal(da,
           (DMDATSRHSJacobianLocal)FormRHSJacobianLocal,&user); CHKERRQ(ierr);
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
    if (dump) {
        PetscViewer  viewer;
        char         filename[PETSC_MAX_PATH_LEN] = "";
        ierr = sprintf(filename,"%s_initial.dat",fileroot);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "writing PETSC binary file %s ...\n",filename); CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,
            FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
        ierr = VecView(u,viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }
    ierr = TSSolve(ts,u); CHKERRQ(ierr);
    if (dump) {
        PetscViewer  viewer;
        char         filename[PETSC_MAX_PATH_LEN] = "";
        ierr = sprintf(filename,"%s_final.dat",fileroot);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "writing PETSC binary file %s ...\n",filename); CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,
            FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
        ierr = VecView(u,viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

