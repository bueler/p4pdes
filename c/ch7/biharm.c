static char help[] =
"Solve the linear biharmonic equation in 2D.  Equation is\n"
"  - grad^4 u = f\n"
"equivalently  - u_xxxx - 2 u_xxyy - u_yyyy = f(x,y),  on the unit square\n"
"S=(0,1)^2 subject to homogeneous simply-supported boundary conditions:\n"
"  u = 0,  Lap u = 0\n"
"where Lap u = u_xx + u_yy.  The equation is rewritten as a 2x2 block system\n"
"with SPD Laplacian blocks on the diagonal,\n"
"   / -Lap |   0  \\  / v \\   /  0 \\ \n"
"   |------|------|  |---| = |----| \n"
"   \\  -I  | -Lap /  \\ u /   \\ -f / \n"
"Includes manufactured, polynomial exact solution.  The discretization is\n"
"structured-grid (DMDA) finite differences.  Option -snes_fd_color (essentially)\n"
"required to form the Jacobian.  Recommended preconditioning combines\n"
"Gauss-Seidel-type (i.e. multiplicative) fieldsplit with GMG as the\n"
"preconditioner for the diagonal blocks:\n"
"   -pc_type fieldsplit -pc_fieldsplit_type multiplicative \\\n"
"   -fieldsplit_v_pc_type mg -fieldsplit_u_pc_type mg\n\n";

/*
1. view of solver; actually seems to be multiplicative fieldsplit with 2-level multigrid as preconditioner in each block:
    ./biharm -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -da_refine 1 -fieldsplit_v_pc_type mg -fieldsplit_v_pc_mg_galerkin -fieldsplit_v_pc_mg_levels 2 -fieldsplit_u_pc_type mg -fieldsplit_u_pc_mg_galerkin -fieldsplit_u_pc_mg_levels 2 -snes_view |less
  NOTE:  -fieldsplit_?_ksp_type preonly  is default
  BUG: removing "galerkin"s causes memory corruption errors

2. optimal; 2 KSP iterations on all grids up to 2049^2:
    for LEV in 5 6 7 8 9 10; do ./biharm -ksp_converged_reason -pc_type fieldsplit -da_refine $LEV -fieldsplit_v_pc_type mg -fieldsplit_v_pc_mg_galerkin -fieldsplit_v_pc_mg_levels $LEV -fieldsplit_u_pc_type mg -fieldsplit_u_pc_mg_galerkin -fieldsplit_u_pc_mg_levels $LEV -fieldsplit_v_mg_levels_ksp_type richardson -fieldsplit_u_mg_levels_ksp_type richardson; done
QUESTIONS:
  a) why need to set mg_levels (-fieldsplit_?_pc_mg_levels $LEV)?
  b) BUG: fails without galerkin

3. one can do "monolithic" GMG too; 3 KSP iterations on all grids up to 2049^2:
    for LEV in 5 6 7 8 9 10; do ./biharm -ksp_converged_reason -pc_type mg -da_refine $LEV -mg_levels_ksp_type richardson; done

4. GAMG works fine too
*/
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
    double         normv, normu, errv, erru;
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
    ierr = DMDASetFieldName(da,0,"v"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"u"); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetType(snes,SNESKSPONLY); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&w_initial); CHKERRQ(ierr);
    ierr = VecSet(w_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,w_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&w_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&w); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

    ierr = VecDuplicate(w,&w_exact); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,w_exact,&aW); CHKERRQ(ierr);
    ierr = FormExactWLocal(&info,aW,&user); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,w_exact,&aW); CHKERRQ(ierr);
    ierr = VecStrideNorm(w_exact,0,NORM_INFINITY,&normv); CHKERRQ(ierr);
    ierr = VecStrideNorm(w_exact,1,NORM_INFINITY,&normu); CHKERRQ(ierr);
    ierr = VecAXPY(w,-1.0,w_exact); CHKERRQ(ierr);
    ierr = VecStrideNorm(w,0,NORM_INFINITY,&errv); CHKERRQ(ierr);
    ierr = VecStrideNorm(w,1,NORM_INFINITY,&erru); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid ...\n"
        "  errors |v-vex|_inf/|vex|_inf = %.5e, |u-uex|_inf/|uex|_inf = %.5e\n",
        info.mx,info.my,errv/normv,erru/normu); CHKERRQ(ierr);

    ierr = VecDestroy(&w_exact); CHKERRQ(ierr);
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

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field **aW,
                                 Field **FF, BiharmCtx *user) {
    PetscErrorCode ierr;
    int        i, j;
    double     xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
               ve, vw, vn, vs, ue, uw, un, us;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    darea = hx * hy;               // multiply FD equations by this
    scx = hy / hx;
    scy = hx / hy;
    scdiag = 2.0 * (scx + scy);    // diagonal scaling
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                FF[j][i].v = scdiag * aW[j][i].v;
                FF[j][i].u = scdiag * aW[j][i].u;
            } else {
                ve = (i+1 == info->mx-1) ? 0.0 : aW[j][i+1].v;
                vw = (i-1 == 0)          ? 0.0 : aW[j][i-1].v;
                vn = (j+1 == info->my-1) ? 0.0 : aW[j+1][i].v;
                vs = (j-1 == 0)          ? 0.0 : aW[j-1][i].v;
                FF[j][i].v = scdiag * aW[j][i].v - scx * (vw + ve) - scy * (vs + vn)
                             - darea * (*(user->f))(x,y);
                ue = (i+1 == info->mx-1) ? 0.0 : aW[j][i+1].u;
                uw = (i-1 == 0)          ? 0.0 : aW[j][i-1].u;
                un = (j+1 == info->my-1) ? 0.0 : aW[j+1][i].u;
                us = (j-1 == 0)          ? 0.0 : aW[j-1][i].u;
                FF[j][i].u = - darea * aW[j][i].v
                             + scdiag * aW[j][i].u - scx * (uw + ue) - scy * (us + un);
            }
        }
    }
    ierr = PetscLogFlops(18.0*info->xm*info->ym);CHKERRQ(ierr);
    return 0;
}

