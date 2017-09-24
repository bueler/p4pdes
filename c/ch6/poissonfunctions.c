#include <petsc.h>
#include "poissonfunctions.h"

// FIXME make all parts work with cx,cy,cz arbitrary; add tests of fish for that

PetscErrorCode Form1DFunctionLocal(DMDALocalInfo *info, double *au,
                                   double *aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int          i;
    double       xmax[1], xmin[1], h, x, ue, uw;
    ierr = DMDAGetBoundingBox(info->da,xmin,xmax); CHKERRQ(ierr);
    h = (xmax[0] - xmin[0]) / (info->mx - 1);
    for (i = info->xs; i < info->xs + info->xm; i++) {
        x = xmin[0] + i * h;
        if (i==0 || i==info->mx-1) {
            aF[i] = (2.0 / h) * (au[i] - user->g_bdry(x,0.0,0.0,user));
        } else {
            ue = (i+1 == info->mx-1) ? user->g_bdry(x+h,0.0,0.0,user) : au[i+1];
            uw = (i-1 == 0)          ? user->g_bdry(x-h,0.0,0.0,user) : au[i-1];
            aF[i] = (2.0 * au[i] - uw - ue) / h
                    - h * user->f_rhs(x,0.0,0.0,user);
        }
    }
    return 0;
}

//STARTFORM2DFUNCTION
PetscErrorCode Form2DFunctionLocal(DMDALocalInfo *info, double **au,
                                   double **aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  xymin[2], xymax[2], hx, hy, scdiag, x, y, ue, uw, un, us;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    scdiag = 2.0 * (hy / hx + hx / hy);    // diagonal scaling
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                aF[j][i] = scdiag * (au[j][i] - user->g_bdry(x,y,0.0,user));
            } else {
                ue = (i+1 == info->mx-1) ? user->g_bdry(x+hx,y,0.0,user)
                                         : au[j][i+1];
                uw = (i-1 == 0)          ? user->g_bdry(x-hx,y,0.0,user)
                                         : au[j][i-1];
                un = (j+1 == info->my-1) ? user->g_bdry(x,y+hy,0.0,user)
                                         : au[j+1][i];
                us = (j-1 == 0)          ? user->g_bdry(x,y-hy,0.0,user)
                                         : au[j-1][i];
                aF[j][i] = scdiag * au[j][i]
                           - hy/hx * (uw + ue) - hx/hy * (us + un)
                           - hx*hy * user->f_rhs(x,y,0.0,user);
            }
        }
    }
    return 0;
}
//ENDFORM2DFUNCTION

PetscErrorCode Form3DFunctionLocal(DMDALocalInfo *info, double ***au,
                                   double ***aF, PoissonCtx *user) {
    PetscErrorCode ierr;
    int    i, j, k;
    double xyzmin[3], xyzmax[3], hx, hy, hz, dvol, scx, scy, scz, scdiag,
           x, y, z, ue, uw, un, us, uu, ud;
    ierr = DMDAGetBoundingBox(info->da,xyzmin,xyzmax); CHKERRQ(ierr);
    hx = (xyzmax[0] - xyzmin[0]) / (info->mx - 1);
    hy = (xyzmax[1] - xyzmin[1]) / (info->my - 1);
    hz = (xyzmax[2] - xyzmin[2]) / (info->mz - 1);
    dvol = hx * hy * hz;
    scx = dvol / (hx*hx);
    scy = dvol / (hy*hy);
    scz = dvol / (hz*hz);
    scdiag = 2.0 * (scx + scy + scz);
    for (k = info->zs; k < info->zs + info->zm; k++) {
        z = xyzmin[2] + k * hz;
        for (j = info->ys; j < info->ys + info->ym; j++) {
            y = xyzmin[1] + j * hy;
            for (i = info->xs; i < info->xs + info->xm; i++) {
                x = xyzmin[0] + i * hx;
                if (   i==0 || i==info->mx-1
                    || j==0 || j==info->my-1
                    || k==0 || k==info->mz-1) {
                    aF[k][j][i] = scdiag * (au[k][j][i] - user->g_bdry(x,y,z,user));
                } else {
                    ue = (i+1 == info->mx-1) ? user->g_bdry(x+hx,y,z,user)
                                             : au[k][j][i+1];
                    uw = (i-1 == 0)          ? user->g_bdry(x-hx,y,z,user)
                                             : au[k][j][i-1];
                    un = (j+1 == info->my-1) ? user->g_bdry(x,y+hy,z,user)
                                             : au[k][j+1][i];
                    us = (j-1 == 0)          ? user->g_bdry(x,y-hy,z,user)
                                             : au[k][j-1][i];
                    uu = (k+1 == info->mz-1) ? user->g_bdry(x,y,z+hz,user)
                                             : au[k+1][j][i];
                    ud = (k-1 == 0)          ? user->g_bdry(x,y,z-hz,user)
                                             : au[k-1][j][i];
                    aF[k][j][i] = scdiag * au[k][j][i]
                                  - scx * (uw + ue) - scy * (us + un) - scz * (uu + ud)
                                  - dvol * user->f_rhs(x,y,z,user);
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
    double       xmin[1], xmax[1], h, v[3];
    MatStencil   col[3],row;

    ierr = DMDAGetBoundingBox(info->da,xmin,xmax); CHKERRQ(ierr);
    h = (xmax[0] - xmin[0]) / (info->mx - 1);
    for (i = info->xs; i < info->xs+info->xm; i++) {
        row.i = i;
        col[0].i = i;
        ncols = 1;
        if (i==0 || i==info->mx-1) {
            v[0] = 2.0 / h;
        } else {
            v[0] = 2.0 / h;
            if (i-1 > 0) {
                col[ncols].i = i-1;  v[ncols++] = - 1.0 / h;
            }
            if (i+1 < info->mx-1) {
                col[ncols].i = i+1;  v[ncols++] = - 1.0 / h;
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
    double       xymin[2], xymax[2], hx, hy, hyhx, hxhy, scdiag, v[5];
    int          i,j,ncols;
    MatStencil   col[5],row;

    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    hyhx = hy / hx;
    hxhy = hx / hy;
    scdiag = 2.0 * (hyhx + hxhy);
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;
        col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;
            col[0].i = i;
            ncols = 1;
            v[0] = scdiag;
            if (i>0 && i<info->mx-1 && j>0 && j<info->my-1) {
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
    double       xyzmin[3], xyzmax[3], hx, hy, hz, dvol, scx, scy, scz, scdiag, v[7];
    int          i,j,k,ncols;
    MatStencil   col[7],row;

    ierr = DMDAGetBoundingBox(info->da,xyzmin,xyzmax); CHKERRQ(ierr);
    hx = (xyzmax[0] - xyzmin[0]) / (info->mx - 1);
    hy = (xyzmax[1] - xyzmin[1]) / (info->my - 1);
    hz = (xyzmax[2] - xyzmin[2]) / (info->mz - 1);
    dvol = hx * hy * hz;
    scx = dvol / (hx*hx);
    scy = dvol / (hy*hy);
    scz = dvol / (hz*hz);
    scdiag = 2.0 * (scx + scy + scz);
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
                v[0] = scdiag;
                if (i>0 && i<info->mx-1 && j>0 && j<info->my-1 && k>0 && k<info->mz-1) {
                    if (i-1 > 0) {
                        col[ncols].k = k;    col[ncols].j = j;    col[ncols].i = i-1;
                        v[ncols++] = - scx;
                    }
                    if (i+1 < info->mx-1) {
                        col[ncols].k = k;    col[ncols].j = j;    col[ncols].i = i+1;
                        v[ncols++] = - scx;
                    }
                    if (j-1 > 0) {
                        col[ncols].k = k;    col[ncols].j = j-1;  col[ncols].i = i;
                        v[ncols++] = - scy;
                    }
                    if (j+1 < info->my-1) {
                        col[ncols].k = k;    col[ncols].j = j+1;  col[ncols].i = i;
                        v[ncols++] = - scy;
                    }
                    if (k-1 > 0) {
                        col[ncols].k = k-1;  col[ncols].j = j;    col[ncols].i = i;
                        v[ncols++] = - scz;
                    }
                    if (k+1 < info->mz-1) {
                        col[ncols].k = k+1;  col[ncols].j = j;    col[ncols].i = i;
                        v[ncols++] = - scz;
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

