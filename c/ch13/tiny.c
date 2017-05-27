static const char help[] =
"Try out DMPlex by building the tiny triangular mesh in the DMPlex part\n"
"of the PETSc User Manual\n\n";

#include <petsc.h>

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM          dmplex;
  int         points[] = {0,1,6,7,8,9,10},
              sizes[]  = {3,3,2,2,2,2, 2},
              cone[]   = {6,7,8,  7,9,10,  2,3,  3,4,  4,2,  4,5,  5,3},
              j, m, start, end;
  const char* names[] = {"nodes",  // or "vertices"
                         "edges",
                         "cells"}; // or "elements"

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = DMPlexCreate(PETSC_COMM_WORLD,&dmplex); CHKERRQ(ierr);
  ierr = DMPlexSetChart(dmplex, 0, 11); CHKERRQ(ierr);
  for (j = 0; j < (int)(sizeof(points) / sizeof(int)); j++) {
     ierr = DMPlexSetConeSize(dmplex, points[j], sizes[j]); CHKERRQ(ierr);
  }
  ierr = DMSetUp(dmplex);
  m = 0;
  for (j = 0; j < (int)(sizeof(points) / sizeof(int)); j++) {
     ierr = DMPlexSetCone(dmplex, points[j], &(cone[m])); CHKERRQ(ierr);
     m += sizes[j];
  }
  ierr = DMPlexSymmetrize(dmplex); CHKERRQ(ierr);  // Symmetrize must be before Stratify!
  ierr = DMPlexStratify(dmplex); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmplex); CHKERRQ(ierr);
  //ierr = DMView(dmplex,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  for (m = 0; m < 3; m++) {
      ierr = DMPlexGetDepthStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s are %d,...,%d\n",names[m],start,end-1); CHKERRQ(ierr);
      //ierr = DMPlexGetHeightStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
      //ierr = PetscPrintf(PETSC_COMM_WORLD,"%s are %d,...,%d\n",names[2-m],start,end-1); CHKERRQ(ierr);
  }
  DMDestroy(&dmplex);
  PetscFinalize();
  return 0;
}

