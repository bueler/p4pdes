//START
static char help[] = "Solve a 4x4 linear system using Vec, Mat, and KSP.\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Vec    x, b;
  Mat    A;
  KSP    ksp;
  int    rank, i, j[4] = {0, 1, 2, 3};                // j = column indices
  double ab[4] = {7.0, 1.0, 1.0, 3.0},          // entries of vector b
         aA[4][4] = {{ 1.0,  2.0,  3.0,  0.0},  // entries of matrix A
                     { 2.0,  1.0, -2.0, -3.0},
                     {-1.0,  1.0,  1.0,  0.0},
                     { 0.0,  1.0,  1.0, -1.0}};

  PetscInitialize(&argc,&args,NULL,help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,4); CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  if (rank == 0) {
      ierr = VecSetValues(b,4,j,ab,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
//  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);  //STRIP

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);
  if (rank == 0) {
      for (i=0; i<4; i++) {   // set entries one row at a time
          ierr = MatSetValues(A,1,&i,4,j,aA[i],INSERT_VALUES); CHKERRQ(ierr);
      }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
//  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);  //STRIP

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
//END

