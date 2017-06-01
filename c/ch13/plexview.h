#ifndef PLEXVIEW_H_
#define PLEXVIEW_H_

/* Homemade view of DMPlex, coordinate PetscSection, and vertex coordinate Vec
according to options:
    -plex_view_ranges
    -plex_view_ranges -plex_view_use_height
    -plex_view_cell_cones
    -plex_view_vertex_supports
    -plex_view_coords                               */
PetscErrorCode PlexViewFromOptions(DM plex);

PetscErrorCode PlexViewRanges(DM plex, PetscBool use_height);

/* viewing cell cones in 2D:
     PlexViewFans(dmplex,2,2,1)
viewing vertex supports:
     PlexViewFans(dmplex,2,0,1)   */
PetscErrorCode PlexViewFans(DM plex, int dim, int basestrata, int targetstrata);

PetscErrorCode VecViewLocalStdout(Vec v, MPI_Comm gcomm);

#endif

