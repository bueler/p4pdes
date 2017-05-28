static const char help[] =
"Build a tiny triangular mesh using DMPlex.  Option prefix tny_.\n\n";
// from two-triangle mesh in PETSc User's Manual

/* compare these views:
$ ./tiny -dm_view
$ ./tiny -tny_print_points
$ ./tiny -tny_print_points -tny_use_height
*/

#include <petsc.h>

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM          dmplex;
  int         points[] = {0,1,6,7,8,9,10},
              sizes[]  = {3,3,2,2,2,2, 2},
              cone[]   = {6,7,8,  7,9,10,  2,3,  3,4,  4,2,  4,5,  5,3},
              j, m;
  PetscBool   print = PETSC_FALSE,
              height = PETSC_FALSE;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tny_", "options for tiny", "");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_points", "print indices for cells,edges,nodes",
                          "tiny.c", print, &print, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_height", "use Height instead of Depth when printing",
                          "tiny.c", height, &height, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  // create the DMPlex "by hand" following user manual
  // ... probably equivalent to DMPlexCreateFromDAG() or DMPlexCreateFromCellList()
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
  ierr = DMPlexSymmetrize(dmplex); CHKERRQ(ierr);  // both are required and Symmetrize must precede Stratify
  ierr = DMPlexStratify(dmplex); CHKERRQ(ierr);
  ierr = DMSetDimension(dmplex, 2); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmplex); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmplex, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmplex, NULL, "-dm_view");CHKERRQ(ierr);

  // show point indices by stratum
  if (print) {
      int         start, end;
      const char* names[] = {"nodes",  // or "vertices"
                             "edges",
                             "cells"}; // or "elements"
      ierr = DMPlexGetChart(dmplex,&start,&end); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"chart has point indices %d,...,%d\n",
                         start,end-1); CHKERRQ(ierr);
      if (height) {
          for (m = 0; m < 3; m++) {
              ierr = DMPlexGetHeightStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD,"    height %d: %s are %d,...,%d\n",
                                 m,names[2-m],start,end-1); CHKERRQ(ierr);
          }
      } else {
          for (m = 0; m < 3; m++) {
              ierr = DMPlexGetDepthStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD,"    depth %d: %s are %d,...,%d\n",
                                 m,names[m],start,end-1); CHKERRQ(ierr);
          }
      }
  }

  DMDestroy(&dmplex);
  return PetscFinalize();
}

