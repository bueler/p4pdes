#ifndef ICE_H_
#define ICE_H_

// a simple parameterized model for the climatic mass balance (CMB), i.e.
// the annual balance of snowfall minus melt, treated as season-less

typedef struct {
    double    ela,        // equilibrium line altitude (m)
              holdelev,   // hold CMB above this elevation (m)
              zgrad,      // vertical derivative (gradient) of CMB (s^-1)
              initmagic;  // constant used to multiply CMB for initial H
} CMBCtx;

PetscErrorCode CMBCtxSetFromOptions(CMBCtx *cmb, double secpera) {
    PetscErrorCode ierr;
    PetscBool      set;
    cmb->ela        = 2000.0; // m
    cmb->zgrad      = 0.004; // a^-1
    cmb->holdelev   = 2250.0; // a^-1
    cmb->initmagic  = 1000.0 * secpera; // s
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"cmb_",
        "options to climatic mass balance (CMB) model, if used","");CHKERRQ(ierr);
    ierr = PetscOptionsReal(
        "-ela", "equilibrium line altitude; in m",
        "ice.h",cmb->ela,&cmb->ela,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal(
        "-holdelev", "hold CMB above this elevation; in m",
        "ice.h",cmb->holdelev,&cmb->holdelev,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal(
        "-initmagic", "used to multiply CMB to get initial thickness iterate; input in a",
        "ice.h",cmb->initmagic,&cmb->initmagic,&set);CHKERRQ(ierr);
    if (set)
        cmb->initmagic *= secpera;
    ierr = PetscOptionsReal(
        "-zgrad", "vertical derivative (gradient) of CMB; input in a^-1",
        "ice.h",cmb->zgrad,&cmb->zgrad,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    cmb->zgrad /= secpera;
    PetscFunctionReturn(0);
}

//FIXME achieving more reliable convergence may require this to be more smooth?
double M_CMB(double s, CMBCtx *cmb) {
    if (s <= cmb->holdelev) {
        return cmb->zgrad * (s - cmb->ela);
    } else {
        return cmb->zgrad * (cmb->holdelev - cmb->ela);
    }
}

// initialize by formula based on surface elevation:
//     H(x,y) <- initmagic * max{CMB(s(x,y)), 0.0}
PetscErrorCode ChopScaleInitialHLocal(DMDALocalInfo *info, double **as,
                                      double **aH, CMBCtx *cmb) {
    int             j,k;
    double          M;
    for (k = info->ys; k < info->ys + info->ym; k++) {
        for (j = info->xs; j < info->xs + info->xm; j++) {
            M = M_CMB(as[k][j],cmb);   // M <- CMB(s(x,y))
            aH[k][j] =  (M < 0.0) ? 0.0 : M;
            aH[k][j] *= cmb->initmagic;
        }
    }
    PetscFunctionReturn(0);
}

// a simple ice sheet model using the shallow ice approximation

typedef struct {
    double    secpera,    // number of seconds in a year
              g,          // acceleration of gravity
              rho,        // ice density
              n,          // Glen exponent for SIA flux term
              A,          // ice softness
              Gamma;      // overall constant
    PetscBool verif;      // use dome formulas if true
    double    domeL,      // dome solution only used on (0,L) x (0,L)
              domeR,      // radius of dome solution
              domeH0,     // center height of dome solution
              domeCx,     // x-coordinate of center of dome solution
              domeCy;     // y-coordinate ...
    CMBCtx    cmb;  // FIXME  if this is not a pointer, may need flag on whether to use it
} IceCtx;

double DomeCMB(double x, double y, IceCtx *user) {
    const double n  = user->n,
                 pp = 1.0 / n,
                 CC = user->Gamma * PetscPowReal(user->domeH0,2.0*n+2.0)
                        / PetscPowReal(2.0 * user->domeR * (1.0-1.0/n),n),
                 xc = x - user->domeCx,
                 yc = y - user->domeCy;
    double       r, s, tmp1, tmp2;
    
    r = PetscSqrtReal(xc*xc + yc*yc);
    // avoid singularities at center and margin
    if (r < 0.01)
        r = 0.01;
    if (r > user->domeR - 0.01)
        r = user->domeR - 0.01;
    s = r / user->domeR;
    tmp1 = PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp) - 1.0;
    tmp2 = 2.0 * PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp-1.0) * (1.0 - 2.0*s) - 1.0;
    return (CC / r) * PetscPowReal(tmp1,n-1.0) * tmp2;
}

PetscErrorCode DomeThicknessLocal(DMDALocalInfo *info, double **aH, IceCtx *user) {
    PetscErrorCode ierr;
    const double n  = user->n,
                 mm = 1.0 + 1.0 / n,
                 qq = n / (2.0 * n + 2.0),
                 CC = user->domeH0 / PetscPowReal(1.0 - 1.0 / n,qq);
    double       xymin[2], xymax[2], dx, dy, x, y, xc, yc, r, s, tmp;
    int          j, k;
    PetscFunctionBeginUser;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    dx = (xymax[0] - xymin[0]) / (info->mx - 1);
    dy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (k=info->ys; k<info->ys+info->ym; k++) {
        y = xymin[1] + k * dy;
        for (j=info->xs; j<info->xs+info->xm; j++) {
            x = xymin[0] + j * dx;
            xc = x - user->domeCx;
            yc = y - user->domeCy;
            r = PetscSqrtReal(xc*xc + yc*yc);
            // avoid singularities at margin and center
            if (r > user->domeR - 0.01) {
                aH[k][j] = 0.0;
            } else {
                if (r < 0.01)
                    r = 0.01;
                s = r / user->domeR;
                tmp = mm * s - (1.0/n) + PetscPowReal(1.0-s,mm) - PetscPowReal(s,mm);
                aH[k][j] = CC * PetscPowReal(tmp,qq);
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode IceCtxSetFromOptions(IceCtx *user) {
    PetscErrorCode ierr;
    user->secpera = 31556926.0;  // number of seconds in a year
    user->g       = 9.81;        // m/s^2
    user->rho     = 910.0;       // kg/m^3
    user->n       = 3.0;         // Glen exponent
    user->A       = 3.1689e-24;  // 1/(Pa^3 s); EISMINT I value
    user->domeL   = 1800.0e3;    // m; compare domeR
    user->domeR   = 750.0e3;     // m; radius
    user->domeH0  = 3600.0;      // m; center height
    user->domeCx  = 900.0e3;     // m
    user->domeCy  = 900.0e3;    // m
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ice_",
        "options to ice sheet model, if used","");CHKERRQ(ierr);
    ierr = PetscOptionsReal(
        "-A", "set value of ice softness A in units Pa-3 s-1",
        "ice.h",user->A,&user->A,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal(
        "-n", "value of Glen exponent n",
        "ice.h",user->n,&user->n,NULL);CHKERRQ(ierr);
    if (user->n <= 1.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,
            "ERROR: n = %f not allowed ... n > 1.0 is required\n",user->n);
    }
    ierr = PetscOptionsReal(
        "-rho", "ice density in units kg m3",
        "ice.h",user->rho,&user->rho,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    // derived constant computed after other ice properties are set
    user->Gamma = 2.0 * PetscPowReal(user->rho*user->g,user->n) 
                   * user->A / (user->n+2.0);
    ierr = CMBCtxSetFromOptions(&(user->cmb),user->secpera);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#endif

