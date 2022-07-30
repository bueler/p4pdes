//STARTWHOLE
static char help[] = "Solve a tridiagonal system of arbitrary size.\n"
"Option prefix = tri_.\n";

#include <petsc.h>

int main(int argc,char **args) {
    Vec         x, b, xexact;
    Mat         A;
    KSP         ksp;
    PetscInt    m = 4, i, Istart, Iend, j[3];
    PetscReal   v[3], xval, errnorm;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    PetscOptionsBegin(PETSC_COMM_WORLD,"tri_","options for tri",NULL);
    PetscCall(PetscOptionsInt("-m","dimension of linear system","tri.c",m,&m,NULL));
    PetscOptionsEnd();

    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x,PETSC_DECIDE,m));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x,&b));
    PetscCall(VecDuplicate(x,&xexact));

    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
    PetscCall(MatSetOptionsPrefix(A,"a_"));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (i=Istart; i<Iend; i++) {
        if (i == 0) {
            v[0] = 3.0;  v[1] = -1.0;
            j[0] = 0;    j[1] = 1;
            PetscCall(MatSetValues(A,1,&i,2,j,v,INSERT_VALUES));
        } else {
            v[0] = -1.0;  v[1] = 3.0;  v[2] = -1.0;
            j[0] = i-1;   j[1] = i;    j[2] = i+1;
            if (i == m-1) {
                PetscCall(MatSetValues(A,1,&i,2,j,v,INSERT_VALUES));
            } else {
                PetscCall(MatSetValues(A,1,&i,3,j,v,INSERT_VALUES));
            }
        }
        xval = PetscExpReal(PetscCosReal((double)i));
        PetscCall(VecSetValues(xexact,1,&i,&xval,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(xexact));
    PetscCall(VecAssemblyEnd(xexact));
    PetscCall(MatMult(A,xexact,b));

    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,x));

    PetscCall(VecAXPY(x,-1.0,xexact));
    PetscCall(VecNorm(x,NORM_2,&errnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "error for m = %d system is |x-xexact|_2 = %.1e\n",m,errnorm));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&xexact));
    PetscCall(PetscFinalize());
    return 0;
}
//ENDWHOLE
