# put this fragment in fish.py to give extensive view of DMPlex mesh._plex

PETSc.Sys.syncPrint('rank %d has global coordinates:\n %s' \
                    % (mesh.comm.rank,str(mesh._plex.getCoordinates().array)))
PETSc.Sys.syncPrint('rank %d has local coordinates:\n %s' \
                    % (mesh.comm.rank,str(mesh._plex.getCoordinatesLocal().array)))
PETSc.Sys.syncPrint('rank %d has chart %s' 
                    % (mesh.comm.rank,str(mesh._plex.getChart())))
for depth in range(3):
    st = mesh._plex.getDepthStratum(depth)
    PETSc.Sys.syncPrint('  depth %d stratum: %s' % (depth,str(st)))
    if depth > 0:
        for c in range(st[0],st[1]):
            PETSc.Sys.syncPrint('    cone(%d) = %s' % (c,str(mesh._plex.getCone(c))))
PETSc.Sys.syncFlush(comm=COMM_WORLD)

