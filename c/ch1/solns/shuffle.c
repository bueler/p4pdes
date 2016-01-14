static char help[] = "Randomly permute the integers from 1 .. n.  Shows use of\n"
"PetscRandom and PetscOptionsInt().  Uses the time in seconds as a random\n"
"number generator seed.  Uses an implementation of the Durstenfeld-Fisher-Yates\n"
"shuffle; see en.wikipedia.org/wiki/Fisher-Yates_shuffle\n\n";

#include <petsc.h>
#include <time.h>

int main(int argc, char **args) {
  PetscErrorCode  ierr;
  PetscInt        i, j, n=10;
  PetscReal       v, tmp, *a;
  PetscRandom     r;

  PetscInitialize(&argc,&args,NULL,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for shuffle",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","sets n so we permute integers 1 .. n",
                         "shuffle.c",n,&n,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r); CHKERRQ(ierr);
  ierr = PetscRandomSetType(r,PETSCRAND48); CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,(int)time(NULL)); CHKERRQ(ierr);
  ierr = PetscRandomSeed(r); CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&a); CHKERRQ(ierr);
  for (i=0; i<n; i++) {
      a[i] = i+1;
  }

  // the shuffle
  for (i=n-1; i>0; i--) {
      ierr = PetscRandomGetValueReal(r,&v); CHKERRQ(ierr);
      j = (int)floor(i*v);
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
  }

  // print the result
  for (i=0; i<n; i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%.0f ",a[i]); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);

  PetscRandomDestroy(&r);
  PetscFree(a);
  PetscFinalize();
  return 0;
}
