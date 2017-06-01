#include <petsc.h>
#include "plexview.h"

static const char* names[4][4] = {{"nodes","",     "",     ""},       // dim=0 names
                                  {"nodes","cells","",     ""},       // dim=1 names
                                  {"nodes","edges","cells",""},       // dim=2 names
                                  {"nodes","edges","faces","cells"}}; // dim=3 names

PetscErrorCode PlexViewRanges(DM plex, PetscViewer viewer, PetscBool use_height) {
    PetscErrorCode ierr;
    int         dim, m, start, end;
    MPI_Comm    comm;
    PetscMPIInt rank,size;
    ierr = PetscObjectGetComm((PetscObject)plex,&comm); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
    ierr = DMGetDimension(plex,&dim); CHKERRQ(ierr);
    ierr = DMPlexGetChart(plex,&start,&end); CHKERRQ(ierr);
    if (size > 1) {
        ierr = PetscSynchronizedPrintf(comm,"[rank %d] ",rank); CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(comm,
        "chart for %d-dimensional DMPlex has points %d,...,%d\n",
        dim,start,end-1); CHKERRQ(ierr);
    for (m = 0; m < dim + 1; m++) {
        if (use_height) {
            ierr = DMPlexGetHeightStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscSynchronizedPrintf(comm,
                "    height %d of size %d: %d,...,%d (%s)\n",
                m,end-start,start,end-1,dim < 4 ? names[dim][2-m] : ""); CHKERRQ(ierr);
        } else {
            ierr = DMPlexGetDepthStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscSynchronizedPrintf(comm,
                "    depth=dim %d of size %d: %d,...,%d (%s)\n",
                m,end-start,start,end-1,dim < 4 ? names[dim][m] : ""); CHKERRQ(ierr);
        }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
    return 0;
}

// FIXME changes including: use Synchronized as above and get start,end locally from chart
PetscErrorCode PlexViewFans(DM plex, PetscViewer viewer, PetscBool use_cone,
                            const char* basename, const char* targetname,
                            int start, int end) {
    PetscErrorCode ierr;
    const int *targets;
    int       j, m, size;
    ierr = PetscViewerASCIIPushSynchronized(viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,
        "%s (= %s indices) of each %s:\n",
        use_cone ? "cones" : "supports",targetname,basename); CHKERRQ(ierr);
    for (m = start; m < end; m++) {
        if (use_cone) {
            ierr = DMPlexGetConeSize(plex,m,&size); CHKERRQ(ierr);
            ierr = DMPlexGetCone(plex,m,&targets); CHKERRQ(ierr);
        } else {
            ierr = DMPlexGetSupportSize(plex,m,&size); CHKERRQ(ierr);
            ierr = DMPlexGetSupport(plex,m,&targets); CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,
            "    %s %d: ",basename,m); CHKERRQ(ierr);
        for (j = 0; j < size-1; j++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,
                "%d,",targets[j]); CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,
            "%d\n",targets[size-1]); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopSynchronized(viewer); CHKERRQ(ierr);
    return 0;
}

