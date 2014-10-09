
static char help[] =
"Solve the Poisson equation\n\
  - div(grad u) = f,\n\
with a mix of Dirichlet and Neumann boundary conditions,\n\
  u          = g      on bdry_D Omega,\n\
  grad u . n = gamma  on bdry_N Omega,\n\
using an unstructured mesh FEM method.\n\
This version uses a manufactured solution. (FIXME: make optional)\n\
For a one-process, very coarse grid example do:\n\
     triangle -pqa0.15 square  # generates square.1.{node,ele,poly}\n\
     c2convert -f square.1     # reads square.1.{node,ele,poly}; generates square.1.petsc\n\
     c2poisson -f square.1     # reads square.1.petsc and solves the equation\n\
To see the matrix graphically:\n\
     c2poisson -f square.1 -a_mat_view draw -draw_pause 5\n\n";

#include <petscksp.h>
#include "convenience.h"
#include "readmesh.h"
#include "poissontools.h"

PetscScalar neuexactsoln(PetscScalar x, PetscScalar y) {
  return cos(2.0*PETSC_PI*x) * cos(2.0*PETSC_PI*y);
}

PetscScalar neufsource(PetscScalar x, PetscScalar y) {
  return 8.0 * PETSC_PI * PETSC_PI * neuexactsoln(x,y);
}

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  WORLD = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  MPI_Comm_rank(WORLD,&rank);

  // READ MESH FROM FILE
  Vec      E,     // element data structure
           x, y;  // coords of node
  PetscInt N,     // number of nodes
           K,     // number of elements
           bs;    // block size for elementtype
  char     fname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  ierr = getmeshfile(WORLD, ".petsc", fname, &viewer); CHKERRQ(ierr);
  ierr = readmesh(WORLD, viewer, &E, &x, &y); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = getcheckmeshsizes(WORLD,E,x,y,&N,&K,&bs); CHKERRQ(ierr);

  // GET NODE AND ELEMENT OWNERSHIP RANGES
  PetscInt Istart,Iend,Kstart,Kend;
  ierr = VecGetOwnershipRange(x,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);

  // CREATE MAT AND RHS b
  Mat A;
  Vec b;
  ierr = MatCreate(WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  // FIXME:  instead of next two lines we should preallocate correctly
  ierr = MatSetUp(A); CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
  ierr = VecSet(b,0.0); CHKERRQ(ierr);

  // ASSEMBLE INITIAL STIFFNESS (IGNORING BOUNDARY VALUES)
  ierr = PetscPrintf(WORLD,"  assembling initial stiffness matrix A ...\n"); CHKERRQ(ierr);
  ierr = initassemble(WORLD,E,&neufsource,NULL,bs,Kstart,Kend,A,b); CHKERRQ(ierr);

  // MINIMAL CHECK IS THAT U=1 IS IN KERNEL
  Vec uone, btest;
  PetscScalar normone, normAone;
  ierr = VecDuplicate(b,&uone); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&btest); CHKERRQ(ierr);
  ierr = VecSet(uone,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uone,btest); CHKERRQ(ierr);             // btest = A * uone
  ierr = VecNorm(uone,NORM_2,&normone); CHKERRQ(ierr);
  ierr = VecNorm(btest,NORM_2,&normAone); CHKERRQ(ierr);
  ierr = PetscPrintf(WORLD,"  check I:  |A0 * 1|_2 / |1|_2 = %e   (should be O(eps))\n",
                     normAone/normone); CHKERRQ(ierr);
  ierr = VecDestroy(&uone); CHKERRQ(ierr);
  ierr = VecDestroy(&btest); CHKERRQ(ierr);

  // SOLVE HOMOGENEOUS NEUMANN PROBLEM WITH KNOWN SOLN
  // FIRST EVALUATE EXACT SOLUTION AT NODES
  Vec         uexact;
  PetscScalar uval, *ax, *ay;
  PetscInt    i;
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
  for (i = Istart; i < Iend; i++) {
    uval = neuexactsoln(ax[i-Istart],ay[i-Istart]);
    ierr = VecSetValues(uexact,1,&i,&uval,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
  vecassembly(uexact)
  // NEXT SOLVE SYSTEM
  Vec          u;
  KSP          ksp;
  MatNullSpace nullsp;
  ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
  ierr = KSPCreate(WORLD, &ksp); CHKERRQ(ierr);
  // only constants are in null space:
  ierr = MatNullSpaceCreate(WORLD, PETSC_TRUE, 0, NULL, &nullsp); CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ksp, nullsp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);
  // NOW COMPUTE ERROR
  PetscScalar normuexact, normerror;
  ierr = VecNorm(uexact,NORM_2,&normuexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u := -uexact + u
  ierr = VecNorm(u,NORM_2,&normerror); CHKERRQ(ierr);
  ierr = PetscPrintf(WORLD,"  solving homogenous: |u - uexact|_2 / |uexact|_2 = %e  (should be O(h^2))\n",
                     normerror/normuexact); CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  // CLEAN UP
  MatDestroy(&A);
  VecDestroy(&b);  VecDestroy(&u);  VecDestroy(&uexact);
  VecDestroy(&x);  VecDestroy(&y);  VecDestroy(&E);
  PetscFinalize();
  return 0;
}
