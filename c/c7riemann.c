
static char help[] = "Implement Gudunov method for system of linear conservation laws.\n"
"The problem being solved is FIXME
"\n\n";

#include <math.h>
#include <petscdmda.h>


PetscErrorCode fillsmallmat(const PetscInt d, PetscReal values[d][d], Mat A) {
  PetscErrorCode ierr;
  PetscInt       col[d], row, j;
  for (j = 0; j < d; j++)
    col[j] = j;
  for (row = 0; row < d; row++) {
    ierr = MatSetValues(A,1,&row,d,col,(PetscReal*)values[row],INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,(char*)0,help);

  const PetscInt d = 2;  // d = DOF
  Mat            A, Aminus;
  // these are dense d x d sequential matrices, unrelated to the grid
  //   (each processor owns whole matrix)
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,d,d,d,NULL,&A); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);

  // fill A
  PetscReal val[d][d], c = 3.0;
  val[0][0] = 0.0;  val[0][1] = c;
  val[1][0] = c;    val[1][1] = 0.0;
  ierr = fillsmallmat(2,val,A); CHKERRQ(ierr);

  // fill Aminus; see getAminus.m for computation of Aminus from A
  ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&Aminus); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(Aminus,"Aminus_"); CHKERRQ(ierr);
  val[0][0] = -1.5; val[0][1] = 1.5;
  val[1][0] = 1.5;  val[1][1] = -1.5;
  ierr = fillsmallmat(2,val,Aminus); CHKERRQ(ierr);

  // set up the grid
  DM da;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                      -50,         // override with -da_grid_x or -da_refine
                      d, 1, NULL,  // dof = 1 and stencil width = 1
                      &da); CHKERRQ(ierr);

  // determine grid locations (cell-centered grid)
  DMDALocalInfo  info;
  PetscReal      L = 10.0, dx;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = L / (PetscReal)(info.mx);
  ierr = DMDASetUniformCoordinates(da,dx/2,L-dx/2,-1.0,-1.0,-1.0,-1.0);CHKERRQ(ierr);

  // u = u(t_n), unew = u(t_n+1)
  Vec  u, unew, F;
  ierr = DMCreateLocalVector(da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"solution u"); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"flux F"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&unew);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)unew,"updated solution unew"); CHKERRQ(ierr);
  ierr = VecSet(unew,0.0); CHKERRQ(ierr);

  // at each cell we will need to compute   Fcell = A qleft + Aminus dq
  Vec dq, qleft, tmp, Fcell;
  ierr = VecCreateSeq(PETSC_COMM_WORLD,d,&dq); CHKERRQ(ierr);
  ierr = VecDuplicate(dq,&qleft); CHKERRQ(ierr);
  ierr = VecDuplicate(dq,&tmp); CHKERRQ(ierr);
  ierr = VecDuplicate(dq,&Fcell); CHKERRQ(ierr);

  // view the solution graphically; control with -draw_pause
  PetscViewer viewer;
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"solution u",
              PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewer); CHKERRQ(ierr);

  /* time-stepping loop */
  PetscReal  t = 0.0, tf = 10.0, dt, nu;
  PetscInt   n, NN = 10;
  dt = tf / NN;
  nu = dt / dx;
  for (n = 0; n < NN; ++n) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  time[%3d]=%6g: \n", n, t); CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(da,unew,INSERT_VALUES,u); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,unew,INSERT_VALUES,u); CHKERRQ(ierr);

    ierr = VecView(u,viewer); CHKERRQ(ierr);

    PetscReal  **au, **aunew, **aF;
    PetscInt   j, p;
    ierr = DMDAVecGetArrayDOF(da, u, &au);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOF(da, F, &aF);CHKERRQ(ierr);
    for (j=info.xs; j<info.xs+info.xm; j++) {
        PetscReal *adq, *aqleft, *aFcell;
        ierr = VecGetArray(dq,&adq); CHKERRQ(ierr);
        ierr = VecGetArray(qleft,&aqleft); CHKERRQ(ierr);
        for (p = 0; p < d; p++) {
          adq[p]    = au[j+1][p] - au[j][p];
          aqleft[p] = au[j][p];
        }
        ierr = VecRestoreArray(dq,&adq); CHKERRQ(ierr);
        ierr = VecRestoreArray(qleft,&aqleft); CHKERRQ(ierr);

        // tmp = A qleft
        // Fcell = tmp + Aminus dq
        ierr = MatMult(A,qleft,tmp); CHKERRQ(ierr);
        ierr = MatMultAdd(Aminus,dq,tmp,Fcell); CHKERRQ(ierr);

        ierr = VecGetArray(Fcell,&aFcell); CHKERRQ(ierr);
        for (p = 0; p < d; p++)
          aF[j][p] = aFcell[p];
        ierr = VecRestoreArray(Fcell,&aFcell); CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArrayDOF(da, F, &aF);CHKERRQ(ierr);

    ierr = DMLocalToLocalBegin(da,F,INSERT_VALUES,F); CHKERRQ(ierr);
    ierr = DMLocalToLocalEnd(da,F,INSERT_VALUES,F); CHKERRQ(ierr);

    ierr = DMDAVecGetArrayDOF(da, F, &aF);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOF(da, unew, &aunew);CHKERRQ(ierr);
    for (j=info.xs; j<info.xs+info.xm; j++) {
        for (p = 0; p < d; p++)
          aunew[j][p] = au[j][p] - nu * (aF[j+1][p] - aF[j][p]);
    }
    ierr = DMDAVecRestoreArrayDOF(da, u, &au);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOF(da, F, &aF);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOF(da, unew, &aunew);CHKERRQ(ierr);

    t += dt;
  }

  // clean up
  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = VecDestroy(&unew); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = VecDestroy(&dq); CHKERRQ(ierr);
  ierr = VecDestroy(&qleft); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp); CHKERRQ(ierr);
  ierr = VecDestroy(&Fcell); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = MatDestroy(&Aminus); CHKERRQ(ierr);  
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);  
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}

