static const char help[] =
"Build and explore a tiny triangular mesh using DMPlex.  Option prefix tny_.\n\n";
// from two-triangle mesh in PETSc User's Manual

/* compare these views:
$ ./tiny -dm_view
$ ./tiny -tny_ranges
$ ./tiny -tny_ranges -tny_height
$ ./tiny -tny_vec_view
*/

#include <petsc.h>

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM            dmplex;
    PetscSection  section;
    const int     points = 11,
                  cells = 2,
                  edgeoff = 6,
                  ccone[2][3] = {{6,7,8},
                                 {7,9,10}},
                  econe[5][2] = {{2,3},
                                 {3,4},
                                 {4,2},
                                 {4,5},
                                 {5,3}};
    int           j, m, pstart, pend, nstart, nend, estart, eend;
    PetscBool     ranges = PETSC_FALSE,
                  height = PETSC_FALSE,
                  vec_view = PETSC_FALSE;

    PetscInitialize(&argc,&argv,NULL,help);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tny_", "options for tiny", "");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-height", "use Height instead of Depth when printing points",
                            "tiny.c", height, &height, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ranges", "print point index ranges for cells,edges,nodes",
                            "tiny.c", ranges, &ranges, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vec_view", "print entries of a global vec for P2 elements",
                            "tiny.c", vec_view, &vec_view, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();

    // create the DMPlex "by hand"
    // ... probably equivalent to DMPlexCreateFromDAG() or DMPlexCreateFromCellList()
    ierr = DMPlexCreate(PETSC_COMM_WORLD,&dmplex); CHKERRQ(ierr);
    ierr = DMPlexSetChart(dmplex, 0, points); CHKERRQ(ierr);
    // the points are cells, nodes, edges in that order, and we only set cones for cells and edges
    for (j = 0; j < cells; j++) {
        ierr = DMPlexSetConeSize(dmplex, j, 3); CHKERRQ(ierr);
    }
    for (j = edgeoff; j < points; j++) {
        ierr = DMPlexSetConeSize(dmplex, j, 2); CHKERRQ(ierr);
    }
    ierr = DMSetUp(dmplex);
    for (j = 0; j < cells; j++) {
        ierr = DMPlexSetCone(dmplex, j, ccone[j]); CHKERRQ(ierr);
    }
    for (j = edgeoff; j < points; j++) {
        ierr = DMPlexSetCone(dmplex, j, econe[j-edgeoff]); CHKERRQ(ierr);
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
        const char* names[] = {"nodes",  // or "vertices"
                               "edges",
                               "cells"}; // or "elements" or "triangles"
        ierr = PetscPrintf(PETSC_COMM_WORLD,"chart has point indices %d,...,%d\n",
                           pstart,pend-1); CHKERRQ(ierr);
        for (m = 0; m < 3; m++) {
            int start, end;
            if (height) {
                ierr = DMPlexGetHeightStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
                ierr = PetscPrintf(PETSC_COMM_WORLD,"    height %d: %s are %d,...,%d\n",
                                   m,names[2-m],start,end-1); CHKERRQ(ierr);
            } else {
                ierr = DMPlexGetDepthStratum(dmplex,m,&start,&end); CHKERRQ(ierr);
                ierr = PetscPrintf(PETSC_COMM_WORLD,"    depth (=dim) %d: %s are %d,...,%d\n",
                                   m,names[m],start,end-1); CHKERRQ(ierr);
            }
        }
    }

    // create dofs like P2 elements using PetscSection
    // with 1 dof on each node (depth==0) and 1 dof on each edge (depth==1)
    // [DMPlexCreateSection() seems to do something like the following]
    ierr = DMPlexGetDepthStratum(dmplex, 0, &nstart, &nend); CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dmplex, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&section); CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section, pstart, pend); CHKERRQ(ierr);
    for (j = nstart; j < nend; ++j) {
        ierr = PetscSectionSetDof(section, j, 1); CHKERRQ(ierr);
    }
    for (j = estart; j < eend; ++j) {
        ierr = PetscSectionSetDof(section, j, 1); CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(section); CHKERRQ(ierr);
    ierr = DMSetDefaultSection(dmplex, section); CHKERRQ(ierr);

    // assign values in a global Vec for the section, i.e. on P2 dofs
    // FIXME a more interesting task would be to have an f(x,y), and attach
    // coordinates to the nodes, and evaluate an integral \int_Omega f(x,y) dx dy
    Vec    v;
    double *av;
    int    numpts, *pts = NULL, dof, off;

    ierr = DMGetGlobalVector(dmplex, &v); CHKERRQ(ierr);
    ierr = VecSet(v,0.0); CHKERRQ(ierr);

    // FIXME Vec gets 1.0 for dofs on cell=1  <-- boring
    VecGetArray(v, &av);
    DMPlexGetTransitiveClosure(dmplex, 1, PETSC_TRUE, &numpts, &pts);
    for (j = 0; j < 2 * numpts; j += 2) {  // skip over orientations
        PetscSectionGetDof(section, pts[j], &dof);
        PetscSectionGetOffset(section, pts[j], &off);
        //ierr = PetscPrintf(PETSC_COMM_WORLD,"j=%d: dof=%d, off=%d\n",
        //                   j,dof,off); CHKERRQ(ierr);
        for (m = 0; m < dof; ++m) {
            av[off+m] = 1.0;
        }
    }
    DMPlexRestoreTransitiveClosure(dmplex, 1, PETSC_TRUE, &numpts, &pts);
    VecRestoreArray(v, &av);

    if (vec_view) {
        ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }
    ierr = DMRestoreGlobalVector(dmplex, &v); CHKERRQ(ierr);

    PetscSectionDestroy(&section); DMDestroy(&dmplex);
    return PetscFinalize();
}

