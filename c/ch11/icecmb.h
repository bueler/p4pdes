#ifndef ICECMB_H_
#define ICECMB_H_

// a simple parameterized model for the climatic mass balance (CMB), i.e.
// the annual balance of snowfall minus melt, treated as season-less
typedef struct {
    double ela,   // equilibrium line altitude (m)
           zgrad; // vertical derivative (gradient) of CMB (s^-1)
} CMBModel;

PetscErrorCode SetFromOptions_CMBModel(CMBModel *cmb, const char *optprefix, double secpera) {
  PetscErrorCode ierr;
  cmb->ela   = 2000.0; // m
  cmb->zgrad = 0.001;  // a^-1
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,optprefix,
            "options to climatic mass balance (CMB) model, if used","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-ela", "equilibrium line altitude, in m",
      "cmbmodel.c",cmb->ela,&cmb->ela,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-zgrad", "vertical derivative (gradient) of CMB, in a^-1",
      "cmbmodel.c",cmb->zgrad,&cmb->zgrad,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  cmb->zgrad /= secpera;
  PetscFunctionReturn(0);
}

double M_CMBModel(CMBModel *cmb, double s) {
  return cmb->zgrad * (s - cmb->ela);
}

#endif

