#include <petsc.h>
#include "plexview.h"

static const char* names[4][4] = {{"nodes","",     "",     ""},       // dim=0 names
                                  {"nodes","cells","",     ""},       // dim=1 names
                                  {"nodes","edges","cells",""},       // dim=2 names
                                  {"nodes","edges","faces","cells"}}; // dim=3 names

PetscErrorCode PlexViewRanges(DM plex, PetscViewer viewer, PetscBool use_height) {
    PetscErrorCode ierr;
    int         dim, m, start, end;
    ierr = PetscViewerASCIIPushSynchronized(viewer); CHKERRQ(ierr);
    ierr = DMGetDimension(plex,&dim); CHKERRQ(ierr);
    ierr = DMPlexGetChart(plex,&start,&end); CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,
        "chart for %d-dimensional DMPlex has point indices %d,...,%d\n",
        dim,start,end-1); CHKERRQ(ierr);
    for (m = 0; m < dim + 1; m++) {
        if (use_height) {
            ierr = DMPlexGetHeightStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,
                "    height %d: %d,...,%d (%s)\n",
                m,start,end-1,dim < 4 ? names[dim][2-m] : ""); CHKERRQ(ierr);
        } else {
            ierr = DMPlexGetDepthStratum(plex,m,&start,&end); CHKERRQ(ierr);
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,
                "    depth=dim %d: %d,...,%d (%s)\n",
                m,start,end-1,dim < 4 ? names[dim][m] : ""); CHKERRQ(ierr);
        }
    }
    ierr = PetscViewerASCIIPopSynchronized(viewer); CHKERRQ(ierr);
    return 0;
}


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

