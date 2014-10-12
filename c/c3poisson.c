
static char help[] =
"Solve the Poisson equation\n\
  - div(grad u) = f,\n\
with a mix of Dirichlet and Neumann boundary conditions,\n\
  u          = g      on bdry_D Omega,\n\
  grad u . n = gamma  on bdry_N Omega,\n\
using an unstructured mesh FEM method.\n\
This version uses a manufactured solution. (FIXME: make optional)\n\
For a one-process, coarse grid example do:\n\
     structured.py mesh.1 5    # generates mesh.1.{node,ele,poly}\n\
     c3convert -f mesh.1       # reads mesh.1.{node,ele,poly}; generates mesh.1.petsc\n\
     c3poisson -f mesh.1       # reads mesh.1.petsc and solves the equation\n\
To see the matrix graphically:\n\
     c3poisson -f mesh.1 -a_mat_view draw -draw_pause 5\n\n";

#include <petscksp.h>
#include "convenience.h"
#include "readmesh.h"
#include "poissontools.h"

// for manufactured solution:  this serves as uexact and as g
PetscScalar manufacture_u(PetscScalar x, PetscScalar y) {
  return x * x + y * y * y * y * y;
}

// for manufactured solution:  f = - div(grad u)
PetscScalar manufacture_f(PetscScalar x, PetscScalar y) {
  return - (2.0 + 20.0 * y * y * y);
}

// for -check 2:  we need f(x,y)=1 when checking sum(b)=area
PetscScalar check_f(PetscScalar x, PetscScalar y) {
  return 1.0;
}

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  WORLD = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  MPI_Comm_rank(WORLD,&rank);

  // READ MESH FROM FILE
//GETMESH
  Vec      E,     // element data structure
           x, y;  // coords of node
  PetscInt N,     // number of nodes
           K,     // number of elements
           bs;    // block size for elementtype
  char     fname[PETSC_MAX_PATH_LEN];
  PetscInt check;
  PetscBool checkset;
  PetscViewer viewer;
  ierr = PetscOptionsBegin(WORLD, "", "options for c3poisson", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-check",
                         "check assembly, ignoring Dirichlet conditions:\n"
                         "  1 = check if constants are kernel,\n"
                         "  2 = check if right side sums to area when f=1\n", "", -1,
                         &check, &checkset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if ((checkset == PETSC_TRUE) && (check < 1) && (check > 2)) {  //STRIP
    SETERRQ(WORLD,1,"invalid argument for option -check");  }  //STRIP
  ierr = getmeshfile(WORLD, ".petsc", fname, &viewer); CHKERRQ(ierr);
  ierr = readmesh(WORLD, viewer, &E, &x, &y); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = getcheckmeshsizes(WORLD,E,x,y,&N,&K,&bs); CHKERRQ(ierr);

  // CREATE MAT AND RHS b
  Mat A;
  Vec b;
  ierr = MatCreate(WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
  ierr = VecSet(b,0.0); CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(b,"b_"); CHKERRQ(ierr);
//ENDMATVECCREATE

  ierr = PetscPrintf(WORLD,"  assembling initial stiffness matrix A ...\n"); CHKERRQ(ierr);

//TWOCHECKS
  if (check == 1) {
    // CHECK 1: IS U=constant IN KERNEL?
    Vec         uone;
    PetscScalar normone, normAone;
    ierr = assemble(WORLD,E,NULL,NULL,NULL,A,b); CHKERRQ(ierr);
    ierr = VecDuplicate(b,&uone); CHKERRQ(ierr);
    ierr = VecSet(uone,1.0); CHKERRQ(ierr);
    ierr = MatMult(A,uone,b); CHKERRQ(ierr);             // b = A * uone
    ierr = VecNorm(uone,NORM_2,&normone); CHKERRQ(ierr);
    ierr = VecNorm(b,NORM_2,&normAone); CHKERRQ(ierr);
    ierr = PetscPrintf(WORLD,"  check 1:  are constants in kernel?\n"
                       "    |A * 1|_2 / |1|_2 = %e   (should be O(eps))\n",
                       normAone/normone); CHKERRQ(ierr);
    ierr = VecDestroy(&uone); CHKERRQ(ierr);
  } else if (check == 2) {
    // CHECK 2: DOES b SUM TO AREA OF REGION?
    PetscScalar bsum;
    ierr = assemble(WORLD,E,&check_f,NULL,NULL,A,b); CHKERRQ(ierr);
    ierr = VecSum(b,&bsum); CHKERRQ(ierr);
    ierr = PetscPrintf(WORLD,"  check 2:  does right side sum to area if f=1?\n"
                       "    sum(b) = %e   (should be area of region)\n",
                       bsum); CHKERRQ(ierr);
//ENDTWOCHECKS
  } else {
    // SOLVE MANUFACTURED DIRICHLET PROBLEM; SHOULD WORK IF ENTIRE BOUNDARY IS
    //   MARKED AS DIRICHLET
//SOLVEMANU
    Vec         u, uexact;
    KSP         ksp;
    PetscScalar *ax, *ay, uval, normdiff;
    PetscInt    Istart, Iend, i;
    // assemble and solve system
    ierr = assemble(WORLD,E,&manufacture_f,&manufacture_u,NULL,A,b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
    ierr = KSPCreate(WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(u,"u_"); CHKERRQ(ierr);
    // put exact solution in a Vec
    ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(uexact,"uexact_"); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(uexact,&Istart,&Iend); CHKERRQ(ierr);
    ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
    for (i = Istart; i < Iend; i++) {
      uval = manufacture_u(ax[i-Istart],ay[i-Istart]);
      ierr = VecSetValues(uexact,1,&i,&uval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
    vecassembly(uexact)
    vecassembly(u)  // FIXME:  JUST NEEDED TO ATTACH "u_" prefix for viewing?
    // compute error
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u := -uexact + u
    ierr = VecNorm(u,NORM_INFINITY,&normdiff); CHKERRQ(ierr);
    ierr = PetscPrintf(WORLD,"  numerical error:\n"
                       "    |u - uexact|_inf = %e   (should be O(h^2))\n",
                       normdiff); CHKERRQ(ierr);
    ierr = VecDestroy(&uexact); CHKERRQ(ierr);
  }
//ENDSOLVEMANU

  MatDestroy(&A);  VecDestroy(&b);
  VecDestroy(&x);  VecDestroy(&y);  VecDestroy(&E);
  PetscFinalize();
  return 0;
}
