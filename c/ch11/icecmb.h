#ifndef ICECMB_H_
#define ICECMB_H_

// a simple parameterized model for the climatic mass balance (CMB), i.e.
// the annual balance of snowfall minus melt, treated as season-less
typedef struct {
    double secpera,
           ela,        // equilibrium line altitude (m)
           zgradabove, // vertical derivative (gradient) of CMB (s^-1), above ela
           zgradbelow, // vertical derivative (gradient) of CMB (s^-1), below ela
           initmagic;  // constant used to multiply CMB for initial H
} CMBModel;

PetscErrorCode SetFromOptions_CMBModel(CMBModel *cmb, const char *optprefix, double secpera) {
  PetscErrorCode ierr;
  PetscBool      set;
  cmb->secpera    = 31556926.0;  // number of seconds in a year
  cmb->ela        = 2000.0; // m
  cmb->zgradabove = 0.001; // a^-1
  cmb->zgradbelow = 0.002; // a^-1
  cmb->initmagic  = 1000.0 * cmb->secpera; // s
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,optprefix,
            "options to climatic mass balance (CMB) model, if used","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-ela", "equilibrium line altitude, in m",
      "icecmb.h",cmb->ela,&cmb->ela,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-initmagic", "used to multiply CMB to get initial thickness iterate; input in a",
      "icecmb.h",cmb->initmagic,&cmb->initmagic,&set);CHKERRQ(ierr);
  if (set)   cmb->initmagic *= cmb->secpera;
  ierr = PetscOptionsReal(
      "-zgradabove", "vertical derivative (gradient) of CMB above ela; input in a^-1",
      "icecmb.h",cmb->zgradabove,&cmb->zgradabove,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-zgradbelow", "vertical derivative (gradient) of CMB below ela; input in a^-1",
      "icecmb.h",cmb->zgradbelow,&cmb->zgradbelow,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  cmb->zgradabove /= secpera;
  cmb->zgradbelow /= secpera;
  PetscFunctionReturn(0);
}

double M_CMBModel(CMBModel *cmb, double s) {
  if (s > cmb->ela) {
      return cmb->zgradabove * (s - cmb->ela);
  } else {
      return cmb->zgradbelow * (s - cmb->ela);
  }
}

// initialize by formula based on surface elevation:
//     H(x,y) <- initmagic * max{CMB(s(x,y)), 0.0}
PetscErrorCode ChopScaleInitialHLocal_CMBModel(CMBModel *cmb, DMDALocalInfo *info,
                                               double **as, double **aH) {
  int             j,k;
  double          M;
  for (k = info->ys; k < info->ys + info->ym; k++) {
      for (j = info->xs; j < info->xs + info->xm; j++) {
          M = M_CMBModel(cmb, as[k][j]);   // M <- CMB(s(x,y))
          aH[k][j] =  (M < 0.0) ? 0.0 : M;
          aH[k][j] *= cmb->initmagic;
      }
  }
  PetscFunctionReturn(0);
}
#endif

