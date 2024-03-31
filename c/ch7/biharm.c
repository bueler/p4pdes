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
    PetscReal  v, u;
} Field;

typedef struct {
    PetscReal  (*f)(PetscReal x, PetscReal y);  // right-hand side
} BiharmCtx;

static PetscReal c(PetscReal x) {
    return x*x*x * (1.0-x)*(1.0-x)*(1.0-x);
}

static PetscReal ddc(PetscReal x) {
    return 6.0 * x * (1.0-x) * (1.0 - 5.0 * x + 5.0 * x*x);
}

static PetscReal d4c(PetscReal x) {
    return - 72.0 * (1.0 - 5.0 * x + 5.0 * x*x);
}

static PetscReal u_exact_fcn(PetscReal x, PetscReal y) {
    return c(x) * c(y);
}

static PetscReal lap_u_exact_fcn(PetscReal x, PetscReal y) {
    return - ddc(x) * c(y) - c(x) * ddc(y);  // Lap u = - grad^2 u
}

static PetscReal f_fcn(PetscReal x, PetscReal y) {
    return d4c(x) * c(y) + 2.0 * ddc(x) * ddc(y) + c(x) * d4c(y);  // Lap^2 u = grad^4 u
}

extern PetscErrorCode FormExactWLocal(DMDALocalInfo*, Field**, BiharmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, Field**, Field **FF, BiharmCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, Field**, Mat, Mat, BiharmCtx*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            w, w_initial, w_exact;
    BiharmCtx      user;
    Field          **aW;
    PetscReal      normv, normu, errv, erru;
    DMDALocalInfo  info;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    user.f = &f_fcn;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        3,3,PETSC_DECIDE,PETSC_DECIDE,
                        2,1,              // degrees of freedom, stencil width
                        NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0));
    PetscCall(DMDASetFieldName(da,0,"v"));
    PetscCall(DMDASetFieldName(da,1,"u"));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunctionFn *)FormFunctionLocal,&user));
    PetscCall(DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobianFn *)FormJacobianLocal,&user));
    PetscCall(SNESSetType(snes,SNESKSPONLY));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(DMGetGlobalVector(da,&w_initial));
    PetscCall(VecSet(w_initial,0.0));
    PetscCall(SNESSolve(snes,NULL,w_initial));
    PetscCall(DMRestoreGlobalVector(da,&w_initial));
    PetscCall(DMDestroy(&da));

    PetscCall(SNESGetSolution(snes,&w));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));

    PetscCall(DMCreateGlobalVector(da,&w_exact));
    PetscCall(DMDAVecGetArray(da,w_exact,&aW));
    PetscCall(FormExactWLocal(&info,aW,&user));
    PetscCall(DMDAVecRestoreArray(da,w_exact,&aW));
    PetscCall(VecStrideNorm(w_exact,0,NORM_INFINITY,&normv));
    PetscCall(VecStrideNorm(w_exact,1,NORM_INFINITY,&normu));
    PetscCall(VecAXPY(w,-1.0,w_exact));
    PetscCall(VecStrideNorm(w,0,NORM_INFINITY,&errv));
    PetscCall(VecStrideNorm(w,1,NORM_INFINITY,&erru));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid ...\n"
        "  errors |v-vex|_inf/|vex|_inf = %.5e, |u-uex|_inf/|uex|_inf = %.5e\n",
        info.mx,info.my,errv/normv,erru/normu));

    PetscCall(VecDestroy(&w_exact));
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormExactWLocal(DMDALocalInfo *info, Field **aW, BiharmCtx *user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
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
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
               ve, vw, vn, vs, ue, uw, un, us;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
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
    PetscCall(PetscLogFlops(18.0*info->xm*info->ym));
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, Field **aW,
                                 Mat J, Mat Jpre, BiharmCtx *user) {
    PetscInt     i, j, c, ncol;
    PetscReal    xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, val[6];
    MatStencil   col[6], row;

    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
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
                    PetscCall(MatSetValuesStencil(Jpre,1,&row,1,col,val,INSERT_VALUES));
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
                    PetscCall(MatSetValuesStencil(Jpre,1,&row,ncol,col,val,INSERT_VALUES));
                }
            }
        }
    }

    PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
    if (J != Jpre) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
