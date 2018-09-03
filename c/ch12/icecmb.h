#ifndef ICECMB_H_
#define ICECMB_H_

// a simple parameterized model for the climatic mass balance (CMB), i.e.
// the annual balance of snowfall minus melt, treated as season-less
typedef struct {
    double secpera,
           ela,        // equilibrium line altitude (m)
           holdelev,   // hold CMB above this elevation (m)
           zgrad,      // vertical derivative (gradient) of CMB (s^-1)
           initmagic;  // constant used to multiply CMB for initial H
} CMBModel;

PetscErrorCode SetFromOptions_CMBModel(CMBModel *cmb, double secpera) {
  PetscErrorCode ierr;
  PetscBool      set;
  cmb->secpera    = 31556926.0;  // number of seconds in a year
  cmb->ela        = 2000.0; // m
  cmb->zgrad      = 0.004; // a^-1
  cmb->holdelev   = 2250.0; // a^-1
  cmb->initmagic  = 1000.0 * cmb->secpera; // s
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ice_cmb_",
            "options to climatic mass balance (CMB) model, if used","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-ela", "equilibrium line altitude; in m",
      "icecmb.h",cmb->ela,&cmb->ela,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-holdelev", "hold CMB above this elevation; in m",
      "icecmb.h",cmb->holdelev,&cmb->holdelev,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-initmagic", "used to multiply CMB to get initial thickness iterate; input in a",
      "icecmb.h",cmb->initmagic,&cmb->initmagic,&set);CHKERRQ(ierr);
  if (set)   cmb->initmagic *= cmb->secpera;
  ierr = PetscOptionsReal(
      "-zgrad", "vertical derivative (gradient) of CMB; input in a^-1",
      "icecmb.h",cmb->zgrad,&cmb->zgrad,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  cmb->zgrad /= secpera;
  PetscFunctionReturn(0);
}

//FIXME achieving more reliable convergence may require this to be more smooth?
double M_CMBModel(CMBModel *cmb, double s) {
  if (s <= cmb->holdelev) {
      return cmb->zgrad * (s - cmb->ela);
  } else {
      return cmb->zgrad * (cmb->holdelev - cmb->ela);
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

