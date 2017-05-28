static const char help[] =
"Build and explore a tiny triangular mesh using DMPlex.  Option prefix tny_.\n\n";
// from two-triangle mesh in PETSc User's Manual

/* compare these views:
$ ./tiny -dm_view
$ ./tiny -tny_ranges
$ ./tiny -tny_ranges -tny_height
$ ./tiny -tny_globalvec
*/

#include <petsc.h>

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM            dmplex;
  PetscSection  ps;
  int           points[] = {0,1,6,7,8,9,10},
                sizes[]  = {3,3,2,2,2,2, 2},
                cone[]   = {6,7,8,  7,9,10,  2,3,  3,4,  4,2,  4,5,  5,3},
                j, m, pstart, pend, nstart, nend, estart, eend;
  PetscBool     ranges = PETSC_FALSE,
                height = PETSC_FALSE,
                globalvec = PETSC_FALSE;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tny_", "options for tiny", "");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-height", "use Height instead of Depth when printing points",
                          "tiny.c", height, &height, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ranges", "print point index ranges for cells,edges,nodes",
                          "tiny.c", ranges, &ranges, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-globalvec", "print entries of a global vec for P2 elements",
                          "tiny.c", globalvec, &globalvec, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  // create the DMPlex "by hand"
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
  ierr = PetscObjectSetName((PetscObject) dmplex, "tiny mesh"); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmplex, NULL, "-dm_view"); CHKERRQ(ierr);  // why not enabled by default?
  ierr = DMPlexGetChart(dmplex,&pstart,&pend); CHKERRQ(ierr);

  // optionally print out point index ranges by stratum
  if (ranges) {
      int         start, end;
      const char* names[] = {"nodes",  // or "vertices"
                             "edges",
                             "cells"}; // or "elements" or "triangles"
      ierr = PetscPrintf(PETSC_COMM_WORLD,"chart has point indices %d,...,%d\n",
                         pstart,pend-1); CHKERRQ(ierr);
      if (height) {
          for (m = 0; m < 3; m++) {
              ierr = DMPlexGetHeightStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD,"    height %d: %s are %d,...,%d\n",
                                 m,names[2-m],start,end-1); CHKERRQ(ierr);
          }
      } else {
          for (m = 0; m < 3; m++) {
              ierr = DMPlexGetDepthStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD,"    depth (=dim) %d: %s are %d,...,%d\n",
                                 m,names[m],start,end-1); CHKERRQ(ierr);
          }
      }
  }

  // create dofs like P2 elements using PetscSection
  // with 1 dof on each node (depth==0) and 1 dof on each edge (depth==1)
  ierr = DMPlexGetDepthStratum(dmplex, 0, &nstart, &nend); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmplex, 1, &estart, &eend); CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&ps); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(ps, pstart, pend); CHKERRQ(ierr);
  for (j = nstart; j < nend; ++j) {
      ierr = PetscSectionSetDof(ps, j, 1); CHKERRQ(ierr);
  }
  for (j = estart; j < eend; ++j) {
      ierr = PetscSectionSetDof(ps, j, 1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(ps); CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmplex, ps); CHKERRQ(ierr);

  // optionally create and view a global Vec
  if (globalvec) {
      Vec v;
      ierr = DMGetGlobalVector(dmplex, &v); CHKERRQ(ierr);
      // FIXME do something more interesting like set nodes one value, edges another
      ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmplex, &v); CHKERRQ(ierr);
  }

  PetscSectionDestroy(&ps); DMDestroy(&dmplex);
  return PetscFinalize();
}

