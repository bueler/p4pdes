static char help[] = "Solve the p-laplacian equation in 2D using an objective function.\n\n";

// RUN AS
//   ./plap -snes_fd_function   (no residual = gradient function evaluation)
//   ./plap -snes_fd_color   (no Jacobian = Hessian function evaluation)

#include <petsc.h>

//STARTCONFIGURE
typedef struct {
  PetscReal p, dx, dy;
  PetscBool manufactured;
  Vec       f;
} PLapCtx;

PetscErrorCode Configure(PLapCtx *user) {
  PetscErrorCode ierr;
  user->p = 2.0;
  user->manufactured = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                   NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-manufactured","use manufactured solution (p=2,4 only)",
                   NULL,user->manufactured,&(user->manufactured),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if ((user->manufactured) && (user->p != 2.0) && (user->p != 4.0)) {
      SETERRQ1(PETSC_COMM_WORLD,1,"no manufactured soln for p=%.3f",user->p);
  }
  return 0;
}

PetscErrorCode PrintResult(DMDALocalInfo *info, SNES snes, Vec u, Vec uexact,
                           PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal            unorm, errnorm;
  SNESConvergedReason  reason;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"on %d x %d grid:  ",info->mx,info->my); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
  if (user->manufactured) {
      ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"|u-u_exact|_inf/|u|_inf = %g\n",
                  errnorm/unorm); CHKERRQ(ierr);
  } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",SNESConvergedReasons[reason]); CHKERRQ(ierr);
  }
  return 0;
}
//ENDCONFIGURE

//STARTEXACTF
PetscErrorCode ExactFLocal(DMDALocalInfo *info,
                           PetscReal **uex, PetscReal **f, PLapCtx *user) {
  PetscInt         i,j;
  PetscReal        x,y,s,c,paray,gradsqr;
  const PetscReal  pi2 = PETSC_PI * PETSC_PI;
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = user->dy * j;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = user->dx * i;
      s = sin(2.0*PETSC_PI*x);
      paray = y * (1.0 - y);
      if (user->manufactured) {
          uex[j][i] = s * paray;
          if (user->p == 2.0) {
            f[j][i] = 2.0 * s * (2.0 * pi2 * paray + 1.0);
          } else if (user->p == 4.0) {
            c = cos(2.0*PETSC_PI*x);
            gradsqr = 4.0 * pi2 * c*c * paray*paray
                      + s*s * (1.0-2.0*y)*(1.0-2.0*y);
            f[j][i] = 0.0 * gradsqr;  // FIXME
          } else {
            SETERRQ(PETSC_COMM_WORLD,1,"HOW DID I GET HERE?");
          }
      } else {
        uex[j][i] = NAN;
        f[j][i] = s * paray;
      }
    }
  }
  return 0;
}
//ENDEXACTF

//STARTOBJECTIVE
PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **u,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal      lobj = 0.0;  // FIXME
  ierr = MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,PETSC_COMM_WORLD); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **f, PLapCtx *user) {
  SETERRQ(PETSC_COMM_WORLD,1,"NOT YET IMPLEMENTED");
  return 0;
}
//ENDFUNCTION

//STARTMAIN
int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM                   da;
  SNES                 snes;
  Vec                  u, uexact;
  PetscReal            **auexact, **af;
  PLapCtx              user;
  DMDALocalInfo        info;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = Configure(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  user.dx = 1.0 / (PetscReal)(info.mx-1);
  user.dy = 1.0 / (PetscReal)(info.my-1);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.f));CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,uexact,&auexact); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,user.f,&af); CHKERRQ(ierr);
  ierr = ExactFLocal(&info,auexact,af,&user); CHKERRQ(ierr);
  VecSet(u,1.0);// FIXME: correctly initialize u
  ierr = DMDAVecRestoreArray(da,uexact,&auexact); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,user.f,&af); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = PrintResult(&info,snes,u,uexact,&user); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

