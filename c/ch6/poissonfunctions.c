#include <petsc.h>
#include "poissonfunctions.h"

PetscErrorCode Form1DFunctionLocal(DMDALocalInfo *info, double *au,
                                   double *aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int          i;
    double       xmax[1], xmin[1], hx, x, ue, uw;
    ierr = DMDAGetBoundingBox(info->da,xmin,xmax); CHKERRQ(ierr);
    hx = (xmax[0] - xmin[0]) / (info->mx - 1);
    for (i = info->xs; i < info->xs + info->xm; i++) {
        x = xmin[0] + i * hx;
        if (i==0 || i==info->mx-1) {
            aF[i] = au[i] - user->u_exact(x,0.0,0.0);
        } else {
            ue = (i+1 == info->mx-1) ? user->u_exact(x+hx,0.0,0.0) : au[i+1];
            uw = (i-1 == 0)          ? user->u_exact(x-hx,0.0,0.0) : au[i-1];
            aF[i] = 2.0 * au[i] - uw - ue
                    - hx * hx * user->f_rhs(x,0.0,0.0);
        }
    }
    return 0;
}

PetscErrorCode Form2DFunctionLocal(DMDALocalInfo *info, double **au,
                                   double **aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  xymin[2], xymax[2], hx, hy, x, y, ue, uw, un, us;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                aF[j][i] = au[j][i] - user->u_exact(x,y,0.0);
            } else {
                ue = (i+1 == info->mx-1) ? user->u_exact(x+hx,y,0.0)
                                         : au[j][i+1];
                uw = (i-1 == 0)          ? user->u_exact(x-hx,y,0.0)
                                         : au[j][i-1];
                un = (j+1 == info->my-1) ? user->u_exact(x,y+hy,0.0)
                                         : au[j+1][i];
                us = (j-1 == 0)          ? user->u_exact(x,y-hy,0.0)
                                         : au[j-1][i];
                aF[j][i] = 2.0 * (hy/hx + hx/hy) * au[j][i]
                           - hy/hx * (uw + ue) - hx/hy * (us + un)
                           - hx*hy * user->f_rhs(x,y,0.0);
            }
        }
    }
    return 0;
}

PetscErrorCode Form3DFunctionLocal(DMDALocalInfo *info, double ***au,
                                   double ***aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int    i, j, k;
    double xyzmin[3], xyzmax[3], hx, hy, hz, h, cx, cy, cz, x, y, z,
           ue, uw, un, us, uu, ud;
    ierr = DMDAGetBoundingBox(info->da,xyzmin,xyzmax); CHKERRQ(ierr);
    hx = (xyzmax[0] - xyzmin[0]) / (info->mx - 1);
    hy = (xyzmax[1] - xyzmin[1]) / (info->my - 1);
    hz = (xyzmax[2] - xyzmin[2]) / (info->mz - 1);
    h = pow(hx*hy*hz,1.0/3.0);
    cx = h*h / (hx*hx);
    cy = h*h / (hy*hy);
    cz = h*h / (hz*hz);
    for (k = info->zs; k < info->zs + info->zm; k++) {
        z = xyzmin[2] + k * hz;
        for (j = info->ys; j < info->ys + info->ym; j++) {
            y = xyzmin[1] + j * hy;
            for (i = info->xs; i < info->xs + info->xm; i++) {
                x = xyzmin[0] + i * hx;
                if (   i==0 || i==info->mx-1
                    || j==0 || j==info->my-1
                    || k==0 || k==info->mz-1) {
                    aF[k][j][i] = au[k][j][i] - user->u_exact(x,y,z);
                } else {
                    ue = (i+1 == info->mx-1) ? user->u_exact(x+hx,y,z)
                                             : au[k][j][i+1];
                    uw = (i-1 == 0)          ? user->u_exact(x-hx,y,z)
                                             : au[k][j][i-1];
                    un = (j+1 == info->my-1) ? user->u_exact(x,y+hy,z)
                                             : au[k][j+1][i];
                    us = (j-1 == 0)          ? user->u_exact(x,y-hy,z)
                                             : au[k][j-1][i];
                    uu = (k+1 == info->mz-1) ? user->u_exact(x,y,z+hz)
                                             : au[k+1][j][i];
                    ud = (k-1 == 0)          ? user->u_exact(x,y,z-hz)
                                             : au[k-1][j][i];
                    aF[k][j][i] = 2.0 * (cx + cy + cz) * au[k][j][i]
                                  - cx * (uw + ue) - cy * (us + un) - cz * (uu + ud)
                                  - h*h * user->f_rhs(x,y,z);
                }
            }
        }
    }
    return 0;
}

PetscErrorCode Form1DJacobianLocal(DMDALocalInfo *info, PetscScalar *au,
                                   Mat J, Mat Jpre, PoissonCtx *user) {
    PetscErrorCode  ierr;
    int          i,ncols;
    double       v[3];
    MatStencil   col[3],row;

    for (i = info->xs; i < info->xs+info->xm; i++) {
        row.i = i;
        col[0].i = i;
        ncols = 1;
        if (i==0 || i==info->mx-1) {
            v[0] = 1.0;
        } else {
            v[0] = 2.0;
            if (i-1 > 0) {
                col[ncols].i = i-1;  v[ncols++] = -1.0;
            }
            if (i+1 < info->mx-1) {
                col[ncols].i = i+1;  v[ncols++] = -1.0;
            }
        }
        ierr = MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode Form2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                   Mat J, Mat Jpre, PoissonCtx *user) {
    PetscErrorCode  ierr;
    double       dx, dy, hyhx, hxhy, xymin[2], xymax[2];
    int          i,j,ncols;
    double       v[5];
    MatStencil   col[5],row;

    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    dx = (xymax[0] - xymin[0]) / (info->mx - 1);
    dy = (xymax[1] - xymin[1]) / (info->my - 1);
    hyhx = dy / dx;
    hxhy = dx / dy;
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;
        col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;
            col[0].i = i;
            ncols = 1;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                v[0] = 1.0;
            } else {
                v[0] = 2*(hyhx + hxhy);
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = -hyhx;  }
                if (i+1 < info->mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = -hyhx;  }
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = -hxhy;  }
                if (j+1 < info->my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = -hxhy;  }
            }
            ierr = MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
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

PetscErrorCode Form3DJacobianLocal(DMDALocalInfo *info, PetscScalar ***au,
                                   Mat J, Mat Jpre, PoissonCtx *user) {
    PetscErrorCode  ierr;
    double       dx, dy, dz, h, cx, cy, cz, xyzmin[3], xyzmax[3];
    int          i,j,k,ncols;
    double       v[7];
    MatStencil   col[7],row;

    ierr = DMDAGetBoundingBox(info->da,xyzmin,xyzmax); CHKERRQ(ierr);
    dx = (xyzmax[0] - xyzmin[0]) / (info->mx - 1);
    dy = (xyzmax[1] - xyzmin[1]) / (info->my - 1);
    dz = (xyzmax[2] - xyzmin[2]) / (info->mz - 1);
    h = pow(dx*dy*dz,1.0/3.0),
    cx = h*h / (dx*dx),
    cy = h*h / (dy*dy),
    cz = h*h / (dz*dz);
    for (k = info->zs; k < info->zs+info->zm; k++) {
        row.k = k;
        col[0].k = k;
        for (j = info->ys; j < info->ys+info->ym; j++) {
            row.j = j;
            col[0].j = j;
            for (i = info->xs; i < info->xs+info->xm; i++) {
                row.i = i;
                col[0].i = i;
                ncols = 1;
                if (   i==0 || i==info->mx-1
                    || j==0 || j==info->my-1
                    || k==0 || k==info->mz-1) {
                    v[0] = 1.0;
                } else {
                    v[0] = 2.0 * (cx + cy + cz);
                    if (i-1 > 0) {
                        col[ncols].k = k;    col[ncols].j = j;    col[ncols].i = i-1;
                        v[ncols++] = - cx;
                    }
                    if (i+1 < info->mx-1) {
                        col[ncols].k = k;    col[ncols].j = j;    col[ncols].i = i+1;
                        v[ncols++] = - cx;
                    }
                    if (j-1 > 0) {
                        col[ncols].k = k;    col[ncols].j = j-1;  col[ncols].i = i;
                        v[ncols++] = - cy;
                    }
                    if (j+1 < info->my-1) {
                        col[ncols].k = k;    col[ncols].j = j+1;  col[ncols].i = i;
                        v[ncols++] = - cy;
                    }
                    if (k-1 > 0) {
                        col[ncols].k = k-1;  col[ncols].j = j;    col[ncols].i = i;
                        v[ncols++] = - cz;
                    }
                    if (k+1 < info->mz-1) {
                        col[ncols].k = k+1;  col[ncols].j = j;    col[ncols].i = i;
                        v[ncols++] = - cz;
                    }
                }
                ierr = MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
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

