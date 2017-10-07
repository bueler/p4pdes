#ifndef PLEXVIEW_H_
#define PLEXVIEW_H_

/* homemade view of DMPlex, coordinate PetscSection, and vertex coordinate Vec
according to options:
    -plex_view_points
    -plex_view_points -plex_view_use_height
    -plex_view_cell_cones
    -plex_view_vertex_supports
    -plex_view_coords
    -plex_view_closures_coords
use "-help |grep plex_view" to list at runtime */
PetscErrorCode PlexViewFromOptions(DM plex);

PetscErrorCode PlexViewPointRanges(DM plex, PetscBool use_height);

/* viewing cell cones in 2D:
     PlexViewFans(dmplex,2,2,1)
viewing vertex supports:
     PlexViewFans(dmplex,2,0,1)   */
PetscErrorCode PlexViewFans(DM plex, int dim, int basestrata, int targetstrata);

PetscErrorCode PlexViewCoords(DM plex);

PetscErrorCode PlexViewClosuresCoords(DM plex);

/* for a local Vec, with components on each rank in gcomm, view each local
part by rank */
PetscErrorCode VecViewLocalStdout(Vec v, MPI_Comm gcomm);

#endif

