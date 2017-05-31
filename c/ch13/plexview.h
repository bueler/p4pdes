#ifndef PLEXVIEW_H_
#define PLEXVIEW_H_

PetscErrorCode PlexViewRanges(DM plex, PetscViewer viewer, PetscBool use_height);

// viewing cell cones:
//     PlexViewFans(dmplex,viewer,PETSC_TRUE,"cell","edge",0,ncell)
// viewing vertex supports:
//     PlexViewFans(dmplex,viewer,PETSC_FALSE,"vertex","edge",ncell,ncell+nvert)
PetscErrorCode PlexViewFans(DM plex, PetscViewer viewer, PetscBool use_cone,
                            const char* basename, const char* targetname,
                            int start, int end);

#endif

