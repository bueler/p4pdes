
static char help[] = "Implement Gudunov method for system of linear conservation laws.\n\n";

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
  PetscReal      L = 10.0, dx;
  DMDALocalInfo  info;

  Vec            u, unew;

  const PetscInt d = 2;
  PetscReal      lambda[d], val[d][d], c = 3.0;
  Mat            A, Aminus, R, Rinv;

  PetscInitialize(&argc,&argv,(char*)0,help);

  // these are dense d x d sequential matrices, unrelated to the grid
  //   (each processor owns whole matrix)
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,d,d,d,NULL,&A); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);

  // FIXME: read this from file?
  // fill A and lambda and R
  /*
  >> c = 3;
  >> A = [0 c; c 0];
  >> [X,D]=eig(A)
  X =
       -0.70711  0.70711
        0.70711  0.70711
  D =
       -3        0
        0        3
  >> R = [-1.0 1.0; 1.0 1.0];
  >> inv(R)
  ans =
       -0.50000  0.50000
        0.50000  0.50000
  */

  lambda[0] = -c;  lambda[1] = c;
  val[0][0] = 0.0;  val[0][1] = c;
  val[1][0] = c;    val[1][1] = 0.0;
  ierr = fillsmallmat(2,val,A); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&R); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&Rinv); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&Aminus); CHKERRQ(ierr);
  val[0][0] = -1.0; val[0][1] = 1.0;
  val[1][0] = 1.0;  val[1][1] = 1.0;
  ierr = fillsmallmat(2,val,R); CHKERRQ(ierr);
  val[0][0] = -0.5; val[0][1] = 0.5;
  val[1][0] = 0.5;  val[1][1] = 0.5;
  ierr = fillsmallmat(2,val,Rinv); CHKERRQ(ierr);

  // set up the grid
  DM da;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                      -50,         // override with -da_grid_x or -da_refine
                      d, 1, NULL,  // dof = 1 and stencil width = 1
                      &da); CHKERRQ(ierr);
  //ierr = DMSetApplicationContext(user.da, &user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = L / (PetscReal)(info.mx);
  // cell-centered grid
  ierr = DMDASetUniformCoordinates(da,dx/2,L-dx/2,-1.0,-1.0,-1.0,-1.0);CHKERRQ(ierr);

  // u = u(t_n), unew = u(t_n+1)
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(u,"u_"); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&unew);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(unew,"unew_"); CHKERRQ(ierr);

  ierr = VecSet(u,0.0); CHKERRQ(ierr);

  /* time-stepping loop */
  {
    PetscReal  t = 0.0, tf = 10.0, dt;
    PetscInt   n, NN = 10;
    dt = tf / NN;
    for (n = 0; n < NN; ++n) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "  time[%3d]=%6g: \n", n, t); CHKERRQ(ierr);
      // do something in here
      ierr = VecCopy(unew, u); CHKERRQ(ierr);
      t += dt;
    }
  }

  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = VecDestroy(&unew); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = MatDestroy(&Aminus); CHKERRQ(ierr);  
  ierr = MatDestroy(&R); CHKERRQ(ierr);  
  ierr = MatDestroy(&Rinv); CHKERRQ(ierr);  
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
//ENDMAIN

