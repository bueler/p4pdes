static char help[] =
"Solve the linear biharmonic equation in 2D.  Equation is\n"
"  Lap^2 u = f\n"
"where Lap = - grad^2 is the positive Laplacian, equivalently\n"
"  u_xxxx + 2 u_xxyy + u_yyyy = f(x,y)\n"
"Domain is unit square  S = (0,1)^2.   Boundary conditions are homogeneous\n"
"simply-supported:  u = 0,  Lap u = 0.  The equation is rewritten as a\n"
"2x2 block system with SPD Laplacian blocks on the diagonal:\n"
"   | Lap |  0  |  | v |   | f | \n"
"   |-----|-----|  |---| = |---| \n"
"   | -I  | Lap |  | u |   | 0 | \n"
"Includes manufactured, polynomial exact solution.  The discretization is\n"
"structured-grid (DMDA) finite differences.  Includes analytical Jacobian.\n"
"Recommended preconditioning combines fieldsplit:\n"
"   -pc_type fieldsplit -pc_fieldsplit_type multiplicative|additive \n"
"with multigrid as the preconditioner for the diagonal blocks:\n"
"   -fieldsplit_v_pc_type mg|gamg -fieldsplit_u_pc_type mg|gamg\n"
"(GMG requires setting levels and Galerkin coarsening.)  One can also do\n"
"monolithic multigrid (-pc_type mg|gamg).\n\n";

#include <petsc.h>

typedef struct {
    double  v, u;
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
    return - ddc(x) * c(y) - c(x) * ddc(y);  // Lap u = - grad^2 u
}

static double f_fcn(double x, double y) {
    return d4c(x) * c(y) + 2.0 * ddc(x) * ddc(y) + c(x) * d4c(y);  // Lap^2 u = grad^4 u
}

extern PetscErrorCode FormExactWLocal(DMDALocalInfo*, Field**, BiharmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, Field**, Field **FF, BiharmCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, Field**, Mat, Mat, BiharmCtx*);

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
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
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
            aW[j][i].v = lap_u_exact_fcn(x,y);
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

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, Field **aW,
                                 Mat J, Mat Jpre, BiharmCtx *user) {
    PetscErrorCode ierr;
    int          i, j, c, ncol;
    double       xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, val[6];
    MatStencil   col[6], row;

    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    darea = hx * hy;               // multiply FD equations by this
    scx = hy / hx;
    scy = hx / hy;
    scdiag = 2.0 * (scx + scy);    // diagonal scaling
    for (j = info->ys; j < info->ys + info->ym; j++) {
        row.j = j;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            row.i = i;
            for (c = 0; c < 2; c++) { // v,u equations are c=0,1
                row.c = c;
                col[0].c = c;     col[0].i = i;       col[0].j = j;
                val[0] = scdiag;
                if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                    ierr = MatSetValuesStencil(Jpre,1,&row,1,col,val,INSERT_VALUES);
                        CHKERRQ(ierr);
                } else {
                    ncol = 1;
                    if (i+1 < info->mx-1) {
                        col[ncol].c = c;  col[ncol].i = i+1;  col[ncol].j = j;
                        val[ncol++] = -scx;
                    }
                    if (i-1 > 0) {
                        col[ncol].c = c;  col[ncol].i = i-1;  col[ncol].j = j;
                        val[ncol++] = -scx;
                    }
                    if (j+1 < info->my-1) {
                        col[ncol].c = c;  col[ncol].i = i;    col[ncol].j = j+1;
                        val[ncol++] = -scy;
                    }
                    if (j-1 > 0) {
                        col[ncol].c = c;  col[ncol].i = i;    col[ncol].j = j-1;
                        val[ncol++] = -scy;
                    }
                    if (c == 1) {  // u equation;  has off-diagonal block entry
                        col[ncol].c = 0;
                        col[ncol].i = i;  col[ncol].j = j;
                        val[ncol++] = - darea;
                    }
                    ierr = MatSetValuesStencil(Jpre,1,&row,ncol,col,val,INSERT_VALUES);
                        CHKERRQ(ierr);
                }
            }
        }
    }

    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

