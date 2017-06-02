#include <petsc.h>
#include "plexview.h"

PetscErrorCode PlexViewFromOptions(DM plex) {
    PetscErrorCode ierr;
    PetscBool  cell_cones = PETSC_FALSE,
               coords = PETSC_FALSE,
               ranges = PETSC_FALSE,
               use_height = PETSC_FALSE,
               vertex_supports = PETSC_FALSE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "plex_view_", "view options for tiny", "");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-cell_cones", "print cones of each cell",
                            "tiny.c", cell_cones, &cell_cones, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-coords", "print section and local vec for vertex coordinates",
                            "tiny.c", coords, &coords, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ranges", "print point index ranges for vertices,edges,cells",
                            "tiny.c", ranges, &ranges, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_height", "use Height instead of Depth when printing points",
                            "tiny.c", use_height, &use_height, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vertex_supports", "print supports of each vertex",
                            "tiny.c", vertex_supports, &vertex_supports, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (ranges) {
        ierr = PlexViewRanges(plex,use_height); CHKERRQ(ierr);
    }
    if (cell_cones) {
        ierr = PlexViewFans(plex,2,2,1); CHKERRQ(ierr);
    }
    if (vertex_supports) {
        ierr = PlexViewFans(plex,2,0,1); CHKERRQ(ierr);
    }
    if (coords) {
        PetscSection coordSection;
        Vec          coordVec;
        ierr = DMGetCoordinateSection(plex, &coordSection); CHKERRQ(ierr);
        if (coordSection) {
            ierr = PetscSectionView(coordSection,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "[vertex coordinates PetscSection has not been set]\n"); CHKERRQ(ierr);
        }
        ierr = DMGetCoordinatesLocal(plex,&coordVec); CHKERRQ(ierr);
        if (coords) {
            ierr = VecViewLocalStdout(coordVec,PETSC_COMM_WORLD); CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "[vertex coordinates Vec not been set]\n"); CHKERRQ(ierr);
        }
    }
    return 0;
}

static const char* stratanames[4][10] =
                       {{"vertex","",    "",    ""},       // dim=0 names
                        {"vertex","cell","",    ""},       // dim=1 names
                        {"vertex","edge","cell",""},       // dim=2 names
                        {"vertex","edge","face","cell"}};  // dim=3 names

PetscErrorCode PlexViewRanges(DM plex, PetscBool use_height) {
    PetscErrorCode ierr;
    int         dim, m, start, end;
    const char  *plexname;
    MPI_Comm    comm;
    PetscMPIInt rank,size;
    ierr = PetscObjectGetComm((PetscObject)plex,&comm); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
    ierr = DMGetDimension(plex,&dim); CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)plex,&plexname); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"DMPlex %s in %dD:\n",plexname,dim); CHKERRQ(ierr);
    if (size > 1) {
        ierr = PetscSynchronizedPrintf(comm,"  [rank %d] ",rank); CHKERRQ(ierr);
    }
    ierr = DMPlexGetChart(plex,&start,&end); CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(comm,
        "chart points %d,...,%d\n",
        start,end-1); CHKERRQ(ierr);
    for (m = 0; m < dim + 1; m++) {
        if (use_height) {
            ierr = DMPlexGetHeightStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscSynchronizedPrintf(comm,
                "    height %d of size %d: %d,...,%d (%s)\n",
                m,end-start,start,end-1,dim < 4 ? stratanames[dim][2-m] : ""); CHKERRQ(ierr);
        } else {
            ierr = DMPlexGetDepthStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscSynchronizedPrintf(comm,
                "    depth=dim %d of size %d: %d,...,%d (%s)\n",
                m,end-start,start,end-1,dim < 4 ? stratanames[dim][m] : ""); CHKERRQ(ierr);
        }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode PlexViewFans(DM plex, int dim, int basestrata, int targetstrata) {
    PetscErrorCode ierr;
    const int   *targets;
    int         j, m, start, end, cssize;
    MPI_Comm    comm;
    PetscMPIInt rank,size;
    ierr = PetscObjectGetComm((PetscObject)plex,&comm); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
    if (size > 1) {
        ierr = PetscSynchronizedPrintf(comm,"  [rank %d] ",rank); CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(comm,
        "%s (= %s indices) of each %s:\n",
        (basestrata > targetstrata) ? "cones" : "supports",
        stratanames[dim][targetstrata],stratanames[dim][basestrata]); CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(plex, basestrata, &start, &end); CHKERRQ(ierr);
    for (m = start; m < end; m++) {
        if (basestrata > targetstrata) {
            ierr = DMPlexGetConeSize(plex,m,&cssize); CHKERRQ(ierr);
            ierr = DMPlexGetCone(plex,m,&targets); CHKERRQ(ierr);
        } else {
            ierr = DMPlexGetSupportSize(plex,m,&cssize); CHKERRQ(ierr);
            ierr = DMPlexGetSupport(plex,m,&targets); CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm,
            "    %s %d: ",stratanames[dim][basestrata],m); CHKERRQ(ierr);
        for (j = 0; j < cssize-1; j++) {
            ierr = PetscSynchronizedPrintf(comm,
                "%d,",targets[j]); CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm,
            "%d\n",targets[cssize-1]); CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
    return 0;
}

// for a local Vec, with components on each rank in gcomm, view each local part by rank
PetscErrorCode VecViewLocalStdout(Vec v, MPI_Comm gcomm) {
    PetscErrorCode ierr;
    int         m,locsize;
    PetscMPIInt rank,size;
    double      *av;
    const char  *vecname;
    ierr = MPI_Comm_size(gcomm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(gcomm,&rank); CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)v,&vecname); CHKERRQ(ierr);
    ierr = PetscPrintf(gcomm,"local Vec: %s %d MPI processes\n",
                       vecname,size); CHKERRQ(ierr);
    if (size > 1) {
        ierr = PetscSynchronizedPrintf(gcomm,"[rank %d]:\n",rank); CHKERRQ(ierr);
    }
    ierr = VecGetLocalSize(v,&locsize); CHKERRQ(ierr);
    ierr = VecGetArray(v, &av); CHKERRQ(ierr);
    for (m = 0; m < locsize; m++) {
        ierr = PetscSynchronizedPrintf(gcomm,"%g\n",av[m]); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(v, &av); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(gcomm,PETSC_STDOUT); CHKERRQ(ierr);
    return 0;
}

