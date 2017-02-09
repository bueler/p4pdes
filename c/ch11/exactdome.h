#ifndef EXACTDOME_H_
#define EXACTDOME_H_

// compute (and regularize) radius from center of [0,L] x [0,L]
double radialcoord(double x, double y, AppCtx *user) {
  double xc = x - user->L/2.0, yc = y - user->L/2.0, r;
  r = PetscSqrtReal(xc * xc + yc * yc);
  return r;
}

double DomeCMB(double x, double y, AppCtx *user) {
  const double  domeR  = 750.0e3,  // radius of exact ice sheet (m)
                domeH0 = 3600.0,   // center thickness of exact ice sheet (m)
                n  = user->n_ice,
                pp = 1.0 / n,
                CC = user->Gamma * PetscPowReal(domeH0,2.0*n+2.0)
                         / PetscPowReal(2.0 * domeR * (1.0-1.0/n),n);
  double        r, s, tmp1, tmp2;
  r = radialcoord(x, y, user);
  // avoid singularities at center, margin, resp.
  if (r < 0.01)
      r = 0.01;
  if (r > domeR - 0.01)
      r = domeR - 0.01;
  s = r / domeR;
  tmp1 = PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp) - 1.0;
  tmp2 = 2.0 * PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp-1.0) * (1.0 - 2.0*s) - 1.0;
  return (CC / r) * PetscPowReal(tmp1,n-1.0) * tmp2;
}


PetscErrorCode DomeThicknessLocal(DMDALocalInfo *info, double **aH, AppCtx *user) {
  const double   domeR  = 750.0e3,  // radius of exact ice sheet (m)
                 domeH0 = 3600.0,   // center thickness of exact ice sheet (m)
                 n  = user->n_ice,
                 mm = 1.0 + 1.0 / n,
                 qq = n / (2.0 * n + 2.0),
                 CC = domeH0 / PetscPowReal(1.0 - 1.0 / n,qq),
                 dx = user->L / (double)(info->mx),
                 dy = user->L / (double)(info->my);
  double         x, y, r, s, tmp;
  int            j, k;

  PetscFunctionBeginUser;
  for (k=info->ys; k<info->ys+info->ym; k++) {
      y = k * dy;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          x = j * dx;
          r = radialcoord(x, y, user);
          // avoid singularities at center, margin, resp.
          if (r < 0.01)
              r = 0.01;
          if (r > domeR - 0.01)
              r = domeR - 0.01;
          s = r / domeR;
          if (r < domeR) {
              s = r / domeR;
              tmp = mm * s - (1.0/n) + PetscPowReal(1.0-s,mm) - PetscPowReal(s,mm);
              aH[k][j] = CC * PetscPowReal(tmp,qq);
          } else {
              aH[k][j] = 0.0;
          }
      }
  }
  PetscFunctionReturn(0);
}

#endif

