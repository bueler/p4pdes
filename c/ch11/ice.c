static const char help[] =
"Solves ice sheet problem in 2D:\n"
"    H_t + div (q^x,q^y) = m,\n"
"    (q^x,q^y) = - Gamma H^{n+2} |grad s|^{n-1} grad s + V H,\n"
"where  H(x,y)  is the ice thickness,  b(x,y)  is the bed elevation,\n"
"s(x,y) = H(x,y) + b(x,y)  is surface elevation, and V(x,y) is an imposed\n"
"sliding velocity.  Note  n > 1  and  Gamma = 2 A (rho g)^n / (n+2).\n"
"Domain is a square  [0,L] x [0,L]  with periodic boundary conditions.\n\n"
"Computed by Q1 FVE method (Bueler, 2016) with FD evaluation of Jacobian.\n"
"Uses SNESVI with constraint  H(x,y) >= 0.\n\n";

#include <petsc.h>

typedef struct {
  // grid independent data
  PetscReal L,      // domain is [0,L] x [0,L]
            n,      // Glen exponent for SIA flux term
            g,      // acceleration of gravity
            rho,    // ice density
            secpera,// number of seconds in a year
            A,      // ice softness
            Gamma,  // coefficient for SIA flux term
            eps,    // regularization parameter for n and D
            delta,  // dimensionless regularization for slope in SIA formulas
            lambda, // amount of upwinding; lambda=0 is none and lambda=1 is "full"
            initmagic;// constant, in years, used to multiply CMB fo initial H
  CMBModel  *cmb;
  // the following describe the fine grid:
  DM        da, quadda, sixteenda;
  Vec       b,      // the bed elevation
            m,      // the (time-independent) surface mass balance
            Hexact, // the exact thickness (valid in verification case)
            Hinit;  // initial state
  PetscReal dx, dy, // grid spacings
  PetscInt  Nx, Ny; // grid has Nx x Ny nodes
} AppCtx;


extern PetscErrorCode SetFromOptionsAppCtx(AppCtx*);
extern PetscErrorCode ChopScaleCMBforInitialH(Vec,AppCtx*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode Step(Vec,SNES*,ContinuationScheme*,SNESConvergedReason*,AppCtx*);


int main(int argc,char **argv) {
  PetscErrorCode      ierr;
  SNES                snes;
  Vec                 H;
  AppCtx              user;
  ContinuationScheme  cs;
  CMBModel            cmb;
  DMDALocalInfo       info;
  SNESConvergedReason reason;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SetFromOptionsAppCtx("ice_",&user); CHKERRQ(ierr);
  ierr = SetFromOptionsCMBModel(&cmb,"cmb_",user.secpera);
  user.cmb = &cmb;

  DomeDefaultGrid(&user);

  // this DMDA is used for scalar fields on nodes; cell-centered grid
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      user.Nx,user.Ny,PETSC_DECIDE,PETSC_DECIDE,
                      1, 1,        // dof=1, stencilwidth=1
                      NULL,NULL,&user.da);
  ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
  ierr = DMSetUp(user.da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(user.da, &user);CHKERRQ(ierr);

  // compute grid spacing
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  user.dx = user.L / (PetscReal)(info.mx);
  user.dy = user.L / (PetscReal)(info.my);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.L, 0.0, user.L, 0.0,1.0); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "solving on [0,L] x [0,L] with  L=%.3f km;\n"
             "grid of  %d x %d  points with spacing  dx = %.6f km  and  dy = %.6f km ...\n",
             user.L/1000.0,info.mx,info.my,user.dx/1000.0,user.dy/1000.0);

FIXME

  // this DMDA is used for evaluating fluxes at 4 quadrature points on each element
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      info.mx,info.my,PETSC_DECIDE,PETSC_DECIDE,
                      4, (user.mtrue==PETSC_TRUE) ? 2 : 1,  // dof=4,  stencilwidth=1 or 2
                      NULL,NULL,&user.quadda);
  ierr = DMSetUp(user.quadda); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.quadda, &user);CHKERRQ(ierr);

  // this DMDA is used for evaluating DfluxDl at 4 quadrature points on each
  // elements but with respect to 4 nodal values
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      info.mx,info.my,PETSC_DECIDE,PETSC_DECIDE,
                      16, 1,  // dof=16,  stencilwidth=1 ALWAYS
                              // SETERRQ() in Jacobian protects from use in mtrue=TRUE case
                      NULL,NULL,&user.sixteenda);
  ierr = DMSetUp(user.sixteenda); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.sixteenda, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&H);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"thickness solution H"); CHKERRQ(ierr);

  ierr = VecDuplicate(H,&user.Hinitial); CHKERRQ(ierr);

  ierr = VecDuplicate(H,&user.b); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.b),"bed elevation b"); CHKERRQ(ierr);
  ierr = VecDuplicate(H,&user.m); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.m),"surface mass balance m"); CHKERRQ(ierr);
  ierr = VecDuplicate(H,&user.Hexact); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.Hexact),"exact/observed thickness H"); CHKERRQ(ierr);

  ierr = VecDuplicate(H,&user.ds.Dnodemax); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.ds.Dnodemax),"maximum diffusivity D at node"); CHKERRQ(ierr);
  ierr = VecDuplicate(H,&user.ds.Wmagnodemax); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.ds.Wmagnodemax),"maximum pseudo-velocity magnitude |W| at node"); CHKERRQ(ierr);

  // fill user.[b,m,Hexact] according to 3 choices: data, dome exact soln, JSA exact soln
  if (user.read) {
      if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"reading b, m, Hexact (or Hobserved) from %s ...\n", user.readname);
      ierr = ReadDataVecs(&user); CHKERRQ(ierr);
  } else {
      if (user.dome) {
          if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"generating b, m, Hexact from dome formulas in Bueler (2003) ...\n");
          ierr = VecSet(user.b,0.0); CHKERRQ(ierr);
          ierr = DomeCMB(user.m,&user); CHKERRQ(ierr);
          ierr = DomeExactThickness(user.Hexact,&user); CHKERRQ(ierr);
      } else if (user.bedstep) {
          if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"generating b, m, Hexact from bedrock step formulas in Jarosch et al (2013) ...\n");
          ierr = BedStepBed(user.b,&user); CHKERRQ(ierr);
          ierr = BedStepCMB(user.m,&user); CHKERRQ(ierr);
          ierr = BedStepExactThickness(user.Hexact,&user); CHKERRQ(ierr);
      } else {
          SETERRQ(PETSC_COMM_WORLD,1,"ERROR: one of user.[dome,bedstep] must be TRUE since user.read is FALSE...\n");
      }
  }

  // fill user.Hinitial according to either -mah_readinitial{surface} foo.dat or chop-scale-CMB
  if (user.readinitial) {
      if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"  reading Hinitial from %s ...\n", user.readinitialname);
      ierr = ReadInitialH(&user); CHKERRQ(ierr);
  } else if (user.readinitialsurface) {
      if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"  generating Hinitial by reading surface from %s\n"
                     "    and subtracting bed which was from %s ...\n",
               user.readinitialname, user.readname);
      ierr = GenerateInitialHFromReadSurface(&user); CHKERRQ(ierr);
  } else {
      if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"  generating Hinitial by chop-and-scale of CMB ...\n");
      ierr = ChopScaleCMBforInitialH(user.Hinitial,&user); CHKERRQ(ierr);
      //ierr = VecSet(user.Hinitial,0.0); CHKERRQ(ierr);
  }

  if (user.showdata) {
      ierr = ShowFields(&user); CHKERRQ(ierr);
  }

  // setup local copy of bed
  ierr = DMCreateLocalVector(user.da,&user.bloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user.da,user.b,INSERT_VALUES,user.bloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user.da,user.b,INSERT_VALUES,user.bloc);CHKERRQ(ierr);

  // initialize the SNESVI
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
                (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(user.da,
                (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,    "solving %s by %s method ...\n",
           (user.dtres > 0.0) ? "backward-Euler time-step problems" : "steady-state problem",
           (user.mtrue) ? "true Mahaffy" :
               ((user.lambda == 0.0) ? "M*-without-upwinding" : "M*"));
  if (user.dtres > 0.0) {
      if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"  time-stepping on interval [0.0 a,%.4f a] with initial step %.4f a\n",
               user.T/user.secpera,user.dtres/user.secpera);
  }

  ierr = VecCopy(user.Hinitial,H); CHKERRQ(ierr);

  gettimeofday(&user.starttime, NULL);

  if (user.dtres > 0.0) {  // time-stepping
      // note "Hinitial" is really "H^{l-1}", the solution at the last time step
      PetscReal  tcurrent = 0.0,
                 dtgoal   = user.dtres,
                 epsdt    = 1.0e-4 * dtgoal,
                 dumpdtlast = 0.0;
      while (tcurrent < user.T - epsdt) {
          PetscInt reducecount = 10;
          user.dtres = PetscMin(user.T - tcurrent, dtgoal); // note this exceeds epsdt
          user.dtjac = user.dtres;
          ierr = Step(H,&snes,&cs,&reason,&user); CHKERRQ(ierr);
          while (reason < 0) {
              if (reducecount == 0) {
                  SETERRQ(PETSC_COMM_WORLD,1,"ERROR:  time-step failure ... giving up on reducing time step\n");
              }
              reducecount--;
              ierr = VecCopy(user.Hinitial,H); CHKERRQ(ierr);
              user.dtres /= 2.0;
              user.dtjac = user.dtres;
              ierr = Step(H,&snes,&cs,&reason,&user); CHKERRQ(ierr);
          }
          ierr = VecCopy(H,user.Hinitial); CHKERRQ(ierr);
          tcurrent += user.dtres;
          if (!user.silent) PetscPrintf(PETSC_COMM_WORLD,"t = %.4f a: completed time step of duration %.4f a in interval [0.0 a,%.4f a]\n",
                   tcurrent/user.secpera,user.dtres/user.secpera,user.T/user.secpera);
          if ((user.dumpdt > 0.0) && (tcurrent >= dumpdtlast + user.dumpdt)) {
              Vec  r;
              char name[64];
              int  strerr;
              ierr = SNESGetFunction(snes,&r,NULL,NULL); CHKERRQ(ierr);
              strerr = sprintf(name,"step%.6f.dat",tcurrent/user.secpera);
              if (strerr < 0) {
                  SETERRQ1(PETSC_COMM_WORLD,6,"sprintf() returned %d < 0 ... stopping\n",strerr);
              }
              ierr = DumpToFile(H,r,name,&user); CHKERRQ(ierr);
              dumpdtlast = tcurrent;
          }
      }
  } else { // steady-state
      ierr = Step(H,&snes,&cs,&reason,&user); CHKERRQ(ierr);
      if (reason >= 0) {
          ierr = VecCopy(H,user.Hinitial); CHKERRQ(ierr);
      }
  }

  gettimeofday(&user.endtime, NULL);

  if (user.history) {
      ierr = WriteHistoryFile(H,"history.txt",argc,argv,&user); CHKERRQ(ierr);
  }

  if (user.dump) {
      Vec r;
      ierr = SNESGetFunction(snes,&r,NULL,NULL); CHKERRQ(ierr);
      ierr = DumpToFile(H,r,"unnamed.dat",&user); CHKERRQ(ierr);
  }

  if ((user.averr) || (user.maxerr)) {
      if (reason < 0)
          PetscPrintf(PETSC_COMM_WORLD,"%s\n",SNESConvergedReasons[reason]);
      else {
          PetscReal enorminf, enorm1;
          ierr = GetErrors(H, &user, &enorminf, &enorm1); CHKERRQ(ierr);
          if (user.averr)
              PetscPrintf(PETSC_COMM_WORLD,"%.14e\n",(double)enorm1 / (info.mx * info.my));
          if (user.maxerr)
              PetscPrintf(PETSC_COMM_WORLD,"%.14e\n",(double)enorminf);
      }
  }

  ierr = VecDestroy(&user.bloc);CHKERRQ(ierr);
  ierr = VecDestroy(&user.ds.Dnodemax);CHKERRQ(ierr);
  ierr = VecDestroy(&user.ds.Wmagnodemax);CHKERRQ(ierr);
  ierr = VecDestroy(&user.m);CHKERRQ(ierr);
  ierr = VecDestroy(&user.b);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Hexact);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Hinitial);CHKERRQ(ierr);
  ierr = VecDestroy(&H);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.quadda);CHKERRQ(ierr);
  ierr = DMDestroy(&user.sixteenda);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}


PetscErrorCode SetFromOptionsAppCtx(AppCtx *user) {
  PetscErrorCode ierr;
  PetscBool      domechosen, dtflg, notryset, Tset;
  char           histprefix[512], initHname[512], initsname[512];

  user->n      = 3.0;
  user->g      = 9.81;       // m/s^2
  user->rho    = 910.0;      // kg/m^3
  user->secpera= 31556926.0;
  user->A      = 1.0e-16/user->secpera; // = 3.17e-24  1/(Pa^3 s); EISMINT I value

  user->initmagic = 1000.0;  // a
  user->delta  = 1.0e-4;

  user->lambda = 0.25;  // amount of upwinding; some trial-and-error with bedstep soln; 0.1 gives some Newton convergence difficulties on refined grid (=125m); earlier M* used 0.5

  user->cmb = NULL;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,optprefix,"options to ice","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-A", "set value of ice softness A in units Pa-3 s-1",
      "ice.c",user->A,&user->A,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-delta", "dimensionless regularization for slope in SIA formulas",
      "ice.c",user->delta,&user->delta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-initmagic", "constant, in years, used to multiply CMB to get initial iterate for thickness",
      "ice.c",user->initmagic,&user->initmagic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-lambda", "amount of upwinding; lambda=0 is none and lambda=1 is full",
      "ice.c",user->lambda,&user->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-n", "value of Glen exponent n",
      "ice.c",user->n,&user->n,NULL);CHKERRQ(ierr);
  if (user->n <= 1.0) {
      SETERRQ1(PETSC_COMM_WORLD,11,"ERROR: n = %f not allowed ... n > 1 is required\n",user->n); }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // derived constant computed after n,A get set
  user->Gamma = 2.0 * PetscPowReal(user->rho*user->g,user->n) * user->A / (user->n+2.0);

  PetscFunctionReturn(0);
}


// set initial H by chop & scale CMB
PetscErrorCode ChopScaleCMBforInitialH(Vec Hinitial, AppCtx *user) {
  PetscErrorCode ierr;
  PetscReal      **ab, **am;
  PetscInt       j, k;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  // if CMB m depends on surface elevation, use thickness zero so s=b
  if ((user->cmbmodel) && (user->cmb != NULL)) {
      ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(user->da, user->b, &ab);CHKERRQ(ierr);
      ierr = DMDAVecGetArray(user->da, user->m, &am);CHKERRQ(ierr);
      for (k=info.ys; k<info.ys+info.ym; k++) {
          for (j=info.xs; j<info.xs+info.xm; j++) {
              M_CMBModel(user->cmb,ab[k][j],&(am[k][j]));
          }
      }
      ierr = DMDAVecRestoreArray(user->da, user->b, &ab);CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(user->da, user->m, &am);CHKERRQ(ierr);
  }
  // now do chop and scale
  ierr = VecCopy(user->m,Hinitial); CHKERRQ(ierr);
  ierr = VecTrueChop(Hinitial,0.0); CHKERRQ(ierr);
  ierr = VecScale(Hinitial,user->initmagic * user->secpera); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//  for call-back: tell SNESVI (variational inequality) that we want
//    0.0 <= H < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = VecSet(Xl,0.0); CHKERRQ(ierr);
  ierr = VecSet(Xu,PETSC_INFINITY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// solve the steady-state problem or do one time step:
//   * applies the continuation scheme
//   * applies recovery if in steady-state and if desired
PetscErrorCode Step(Vec H, SNES *snes, ContinuationScheme *cs, SNESConvergedReason *reason, AppCtx *user) {
  PetscErrorCode ierr;
  Vec            Htry;
  PetscInt       m;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(H,&Htry);CHKERRQ(ierr);

  for (m = startCS(cs); m < endCS(cs); m++) {
      user->eps = epsCS(m,cs);
      ierr = VecCopy(H,Htry); CHKERRQ(ierr);
      ierr = SNESAttempt(snes,Htry,m,reason,user);CHKERRQ(ierr);
      if (*reason < 0) {
          if ((user->divergetryagain) && (user->recoverycount == 0) && (user->dtres <= 0.0)) {
              if (!user->silent) PetscPrintf(PETSC_COMM_WORLD,
                  "         turning on steady-state recovery mode (backward Euler time step of %.2f a) ...\n",
                  user->dtrecovery/user->secpera);
              user->dtres = user->dtrecovery;
              user->dtjac = user->dtrecovery;
              user->recoverycount = 1;
              user->goodm = m-1;
              if (!user->silent) PetscPrintf(PETSC_COMM_WORLD,
                  "         trying again ...\n");
              ierr = VecCopy(H,user->Hinitial); CHKERRQ(ierr);
              ierr = VecCopy(H,Htry); CHKERRQ(ierr);
              ierr = SNESAttempt(snes,Htry,m,reason,user);CHKERRQ(ierr);
          }
          if (*reason < 0) {
              if (m>0) // record last successful eps
                  user->eps = epsCS(m-1,cs);
              break;
          }
      } else if (user->recoverycount > 0)
          user->recoverycount++;
      // actions when successful:
      ierr = VecCopy(Htry,H); CHKERRQ(ierr);
      ierr = StdoutReport(H,user); CHKERRQ(ierr);
      if (user->recoverycount == 0)
          user->goodm = m;
  }

  ierr = VecDestroy(&Htry);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

