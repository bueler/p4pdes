//STARTWHOLE
static char help[] = "Solve a 4x4 linear system using KSP.\n";

#include <petsc.h>

int main(int argc,char **args) {
    Vec        x, b;
    Mat        A;
    KSP        ksp;
    PetscInt   i, j[4] = {0, 1, 2, 3};                // j = column index
    PetscReal  ab[4] = {7.0, 1.0, 1.0, 3.0},          // vector entries
               aA[4][4] = {{ 1.0,  2.0,  3.0,  0.0},  // matrix entries
                           { 2.0,  1.0, -2.0, -3.0},
                           {-1.0,  1.0,  1.0,  0.0},
                           { 0.0,  1.0,  1.0, -1.0}};

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b,PETSC_DECIDE,4));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSetValues(b,4,j,ab,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    for (i=0; i<4; i++) {   // set entries one row at a time
        PetscCall(MatSetValues(A,1,&i,4,j,aA[i],INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(VecDuplicate(b,&x));
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(PetscFinalize());
    return 0;
}
//ENDWHOLE
