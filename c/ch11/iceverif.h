#ifndef ICEVERIF_H_
#define ICEVERIF_H_

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

PetscErrorCode HalfarThicknessLocal(DMDALocalInfo *info, double t_in, double **aH, AppCtx *user) {
  const double   halfarR0 = 750.0e3,  // t=t0 radius of exact ice sheet (m)
                 halfarH0 = 3600.0,   // t=t0 center thickness of exact ice sheet (m)
                 halfaralpha = 1.0/9.0,
                 halfarbeta = 1.0/18.0,
                 n = user->n_ice,    // FIXME error if n!= 3
                 q = 1.0 + 1.0 / n,
                 m = n / (2.0*n + 1.0),
                 // for t0, see equation (9) in Bueler et al (2005); normally 422.45 a
                 halfart0 = (halfarbeta/user->Gamma) * PetscPowReal(7.0/4.0,3.0)
                             * (PetscPowReal(halfarR0,4.0) / PetscPowReal(halfarH0,7.0)),
                 dx = user->L / (double)(info->mx),
                 dy = user->L / (double)(info->my);
  double         x, y, r, t, tmp;
  int            j, k;

  PetscFunctionBeginUser;
  t = (t_in + halfart0) / halfart0;   // so t=0 thickness is t0 state in usual Halfar t-axis
  for (k=info->ys; k<info->ys+info->ym; k++) {
      y = k * dy;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          x = j * dx;
          r = radialcoord(x, y, user);
          r /= halfarR0;
          r /= PetscPowReal(t,halfarbeta);
          tmp = PetscMax( 0.0, 1.0 - PetscPowReal(r,q) );
          aH[k][j] = halfarH0 * PetscPowReal(tmp,m) / PetscPowReal(t,halfaralpha);
      }
  }
  PetscFunctionReturn(0);
}

#endif

