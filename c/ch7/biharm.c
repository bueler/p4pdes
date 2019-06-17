static char help[] =
"Solve the linear biharmonic equation in 2D.  Option prefix bh_.\n"
"Equation is\n"
"  - grad^4 u = f\n"
"equivalently\n"
"  - u_xxxx - 2 u_xxyy - u_yyyy = f(x,y),\n"
"on the unit square S=(0,1)^2 subject to simply-supported boundary conditions:\n"
"  u = 0,  Lap u = 0\n"
"where Lap u = u_xx + u_yy.  The equation is rewritten as a 2x2 block system\n"
"with SPD Laplacian blocks on the diagonal,\n"
"   / -Lap |   0   \\  / v \\   / 0 \\ \n"
"   |------|-------|  |---| = |---| \n"
"   \\ - I  | - Lap /  \\ u /   \\-f / \n"
"Includes manufactured, polynomial exact solution.  The discretization is\n"
"structured-grid (DMDA) finite differences.  Option -snes_fd_color (essentially)\n"
"required to form the Jacobian.  Recommended preconditioning combines\n"
"Gauss-Seidel-type (i.e. multiplicative) fieldsplit with GMG as the\n"
"preconditioner for the diagonal blocks:\n"
"   -pc_type fieldsplit -pc_fieldsplit_type multiplicative\n"
"   -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type mg\n"
"   -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type mg\n\n";

#include <petsc.h>

typedef struct {
    double  v,u;  // order matters here
} Field;

typedef struct {
    double  (*f)(double x, double y);  // right-hand side
} BiharmCtx;

static double c(double x) {
    return x*x*x * (1.0-x)*(1.0-x)*(1.0-x);
}

static double ddc(double x) {
    return 6.0 * x * (1.0-x) * (1.0 - 5.0 * x + 5.0 * x*x);
}

static double d4c(double x) {
    return - 72.0 * (1.0 - 5.0 * x + 5.0 * x*x);
}

static double u_exact_fcn(double x, double y) {
    return c(x) * c(y);
}

static double lap_u_exact_fcn(double x, double y) {
    return ddc(x) * c(y) + c(x) * ddc(y);  // is grad^2 u
}

static double f_fcn(double x, double y) {
    return d4c(x) * c(y) + 2.0 * ddc(x) * ddc(y) + c(x) * d4c(y);  // is grad^4 u
}

extern PetscErrorCode FormExactWLocal(DMDALocalInfo*, Field**, BiharmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, Field**,
                                        Field **FF, BiharmCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            w, w_initial, w_exact;
    BiharmCtx      user;
    Field          **aW;
    double         erru, errv;
    DMDALocalInfo  info;

    PetscInitialize(&argc,&argv,NULL,help);
    user.f = &f_fcn;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        3,3,PETSC_DECIDE,PETSC_DECIDE,
                        2,1,              // degrees of freedom, stencil width
                        NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&w_initial); CHKERRQ(ierr);
    ierr = VecSet(w_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,w_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&w_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&w); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&w_exact); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,w_exact,&aW); CHKERRQ(ierr);
    ierr = FormExactWLocal(&info,aW,&user); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,w_exact,&aW); CHKERRQ(ierr);
    ierr = VecAXPY(w,-1.0,w_exact); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&w_exact); CHKERRQ(ierr);
    ierr = VecStrideNorm(w,0,NORM_INFINITY,&errv); CHKERRQ(ierr);
    ierr = VecStrideNorm(w,1,NORM_INFINITY,&erru); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid;  errors |u-uexact|_inf = %.5e, |v-vexact|_inf = %.5e\n",
        info.mx,info.my,erru,errv); CHKERRQ(ierr);

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormExactWLocal(DMDALocalInfo *info, Field **aW, BiharmCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  xymin[2], xymax[2], hx, hy, x, y;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            aW[j][i].u = u_exact_fcn(x,y);
            aW[j][i].v = - lap_u_exact_fcn(x,y);
        }
    }
    return 0;
}

FIXME from here
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PoissonCtx *user) {
    PetscErrorCode ierr;
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    int        i, j;
    double     xymin[2], xymax[2], hx, hy, hxhy, hyhx, x, y,
               ue, uw, un, us, une, use, unw, usw,
               dux, duy, De, Dw, Dn, Ds;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    hxhy = hx / hy;
    hyhx = hy / hx;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            if (j==0 || i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i] - user->g_bdry(x,y,0.0,user);
            } else {
                // assign neighbor values with either boundary condition or
                //     current u at that point (==> symmetric matrix)
                ue  = (i+1 == info->mx-1) ? user->g_bdry(x+hx,y,0.0,user)
                                          : au[j][i+1];
                uw  = (i-1 == 0)          ? user->g_bdry(x-hx,y,0.0,user)
                                          : au[j][i-1];
                un  = (j+1 == info->my-1) ? user->g_bdry(x,y+hy,0.0,user)
                                          : au[j+1][i];
                us  = (j-1 == 0)          ? user->g_bdry(x,y-hy,0.0,user)
                                          : au[j-1][i];
                if (i+1 == info->mx-1 || j+1 == info->my-1) {
                    une = user->g_bdry(x+hx,y+hy,0.0,user);
                } else {
                    une = au[j+1][i+1];
                }
                if (i-1 == 0 || j+1 == info->my-1) {
                    unw = user->g_bdry(x-hx,y+hy,0.0,user);
                } else {
                    unw = au[j+1][i-1];
                }
                if (i+1 == info->mx-1 || j-1 == 0) {
                    use = user->g_bdry(x+hx,y-hy,0.0,user);
                } else {
                    use = au[j-1][i+1];
                }
                if (i-1 == 0 || j-1 == 0) {
                    usw = user->g_bdry(x-hx,y-hy,0.0,user);
                } else {
                    usw = au[j-1][i-1];
                }
                // gradient  (dux,duy)   at east point  (i+1/2,j):
                dux = (ue - au[j][i]) / hx;
                duy = (un + une - us - use) / (4.0 * hy);
                De = DD(dux * dux + duy * duy, mctx->q);
                // ...                   at west point  (i-1/2,j):
                dux = (au[j][i] - uw) / hx;
                duy = (unw + un - usw - us) / (4.0 * hy);
                Dw = DD(dux * dux + duy * duy, mctx->q);
                // ...                  at north point  (i,j+1/2):
                dux = (ue + une - uw - unw) / (4.0 * hx);
                duy = (un - au[j][i]) / hy;
                Dn = DD(dux * dux + duy * duy, mctx->q);
                // ...                  at south point  (i,j-1/2):
                dux = (ue + use - uw - usw) / (4.0 * hx);
                duy = (au[j][i] - us) / hy;
                Ds = DD(dux * dux + duy * duy, mctx->q);
                // evaluate residual
                FF[j][i] = - hyhx * (De * (ue - au[j][i]) - Dw * (au[j][i] - uw))
                           - hxhy * (Dn * (un - au[j][i]) - Ds * (au[j][i] - us));
            }
        }
    }
    return 0;
}

