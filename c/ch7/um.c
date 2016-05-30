#include <petsc.h>
#include "um.h"

PetscErrorCode UMInitialize(UM *mesh) {
    mesh->N = 0;
    mesh->K = 0;
    mesh->P = 0;
    mesh->loc = NULL;
    mesh->e = NULL;
    mesh->bfn = NULL;
    mesh->s = NULL;
    mesh->bfs = NULL;
    return 0;
}

PetscErrorCode UMDestroy(UM *mesh) {
    VecDestroy(&(mesh->loc));
    ISDestroy(&(mesh->e));
    ISDestroy(&(mesh->bfn));
    ISDestroy(&(mesh->s));
    ISDestroy(&(mesh->bfs));
    return 0;
}

PetscErrorCode UMView(UM *mesh, PetscViewer viewer) {
    PetscErrorCode ierr;
    const Node   *aloc;
    int          n, k;
    const int    *ae, *abfn, *as, *abfs;

    ierr = PetscViewerASCIIPushSynchronized(viewer); CHKERRQ(ierr);
    if ((mesh->loc) && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d nodes at (x,y) coordinates:\n",mesh->N); CHKERRQ(ierr);
        ierr = VecGetArrayRead(mesh->loc,(const double **)&aloc); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : (%g,%g)\n",
                               n,aloc[n].x,aloc[n].y); CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(mesh->loc,(const double **)&aloc); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"node coordinates empty/unallocated\n"); CHKERRQ(ierr);
    }
    if ((mesh->e) && (mesh->K > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d elements:\n",mesh->K); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->e,&ae); CHKERRQ(ierr);
        for (k = 0; k < mesh->K; k++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d : %3d %3d %3d\n",
                               k,ae[3*k+0],ae[3*k+1],ae[3*k+2]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->e,&ae); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"element index triples empty/unallocated\n"); CHKERRQ(ierr);
    }
    if ((mesh->bfn) && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary flags at nodes (0 = interior, 1 = boundary, 2 = Dirichlet):\n",mesh->N); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->bfn,&abfn); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %1d\n",
                               n,abfn[n]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->bfn,&abfn); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"boundary flags empty/unallocated\n"); CHKERRQ(ierr);
    }
    if ((mesh->s) && (mesh->P > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary segments:\n",mesh->P); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->s,&as); CHKERRQ(ierr);
        for (n = 0; n < mesh->P; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %3d %3d\n",
                               n,as[2*n+0],as[2*n+1]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->s,&as); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"boundary segment empty/unallocated\n"); CHKERRQ(ierr);
    }
    if ((mesh->bfs) && (mesh->P > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary flags at segments (1 = Neumann, 2 = Dirichlet):\n",mesh->P); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->bfs,&abfs); CHKERRQ(ierr);
        for (n = 0; n < mesh->P; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %d\n",
                               n,abfs[n]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->bfs,&abfs); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"boundary flags at segments empty/unallocated\n"); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopSynchronized(viewer); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMReadNodes(UM *mesh, char *filename) {
    PetscErrorCode ierr;
    int         twoN;
    PetscViewer viewer;
    if (mesh->N > 0) {
        SETERRQ(PETSC_COMM_WORLD,1,"nodes already created?\n");
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&mesh->loc); CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->loc); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(mesh->loc,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(mesh->loc,&twoN); CHKERRQ(ierr);
    if (twoN % 2 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,2,"node locations loaded from %s are not N pairs\n",filename);
    }
    mesh->N = twoN / 2;
    return 0;
}

PetscErrorCode UMCheckElements(UM *mesh) {
    PetscErrorCode ierr;
    const int   *ae;
    int         k, m;
    if ((mesh->K == 0) || (mesh->e == NULL)) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "number of elements unknown; call UMReadElements() first\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "node size unknown so element check impossible; call UMReadNodes() first\n");
    }
    ierr = ISGetIndices(mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < mesh->K; k++) {
        for (m = 0; m < 3; m++) {
            if ((ae[3*k+m] < 0) || (ae[3*k+m] >= mesh->N)) {
                SETERRQ3(PETSC_COMM_WORLD,3,
                   "index e[%d]=%d invalid: not between 0 and N-1=%d\n",
                   3*k+m,ae[3*k+m],mesh->N-1);
            }
        }
        // FIXME: could add check for distinct indices
    }
    ierr = ISRestoreIndices(mesh->e,&ae); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMCheckBoundaryData(UM *mesh) {
    PetscErrorCode ierr;
    const int   *as, *abfn, *abfs;
    int         n, m;
    if (mesh->bfn == NULL) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "boundary flags at nodes not allocated; call UMReadNodes() first\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "node size unknown so boundary flag check impossible; call UMReadNodes() first\n");
    }
    if ((mesh->P == 0) || (mesh->s == NULL)) {
        SETERRQ(PETSC_COMM_WORLD,3,
                "number of boundary segments unknown; call UMReadElements() first\n");
    }
    if (mesh->bfs == NULL) {
        SETERRQ(PETSC_COMM_WORLD,4,
                "boundary flags at segments not allocated; call UMReadElements() first\n");
    }
    ierr = ISGetIndices(mesh->bfn,&abfn); CHKERRQ(ierr);
    for (n = 0; n < mesh->N; n++) {
        switch (abfn[n]) {
            case 0 :
            case 1 :
            case 2 :
                break;
            default :
                SETERRQ2(PETSC_COMM_WORLD,5,
                   "boundary flag bfn[%d]=%d invalid: not in {0,1,2}\n",
                   n,abfn[n]);
        }
    }
    ierr = ISRestoreIndices(mesh->bfn,&abfn); CHKERRQ(ierr);
    ierr = ISGetIndices(mesh->s,&as); CHKERRQ(ierr);
    for (n = 0; n < mesh->P; n++) {
        for (m = 0; m < 2; m++) {
            if ((as[2*n+m] < 0) || (as[2*n+m] >= mesh->N)) {
                SETERRQ3(PETSC_COMM_WORLD,6,
                   "index s[%d]=%d invalid: not between 0 and N-1=%d\n",
                   2*n+m,as[3*n+m],mesh->N-1);
            }
        }
    }
    ierr = ISRestoreIndices(mesh->s,&as); CHKERRQ(ierr);
    ierr = ISGetIndices(mesh->bfs,&abfs); CHKERRQ(ierr);
    for (n = 0; n < mesh->P; n++) {
        switch (abfs[n]) {
            case 0 :
            case 1 :
            case 2 :
                break;
            default :
                SETERRQ2(PETSC_COMM_WORLD,3,
                   "boundary flag bfs[%d]=%d invalid: not in {0,1,2}\n",
                   n,abfs[n]);
        }
    }
    ierr = ISRestoreIndices(mesh->bfs,&abfs); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMReadISs(UM *mesh, char *filename) {
    PetscErrorCode ierr;
    PetscViewer viewer;
    int         n_bfn, n_bfs;
    if ((mesh->K > 0) || (mesh->P > 0)) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "elements already created? ... stopping\n");
    }
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "node coordinates not created ... do that first ... stopping\n");
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    // create and load e
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->e)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->e,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->e,&(mesh->K)); CHKERRQ(ierr);
    if (mesh->K % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,3,
                 "IS e loaded from %s is wrong size for list of element triples\n",filename);
    }
    mesh->K /= 3;
    // create and load bfn
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->bfn)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->bfn,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->bfn,&n_bfn); CHKERRQ(ierr);
    if (n_bfn != mesh->N) {
        SETERRQ1(PETSC_COMM_WORLD,4,
                 "IS bfn loaded from %s is wrong size\n",filename);
    }
    // create and load s
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->s)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->s,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->s,&(mesh->P)); CHKERRQ(ierr);
    if (mesh->P % 2 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,4,
                 "IS s loaded from %s is wrong size for list of segment pairs\n",filename);
    }
    mesh->P /= 2;
    // create and load bfn
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->bfs)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->bfs,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->bfs,&n_bfs); CHKERRQ(ierr);
    if (n_bfs != mesh->P) {
        SETERRQ1(PETSC_COMM_WORLD,4,
                 "IS bfs loaded from %s is wrong size\n",filename);
    }
    // mesh should be complete now
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = UMCheckElements(mesh); CHKERRQ(ierr);
    ierr = UMCheckBoundaryData(mesh); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMGetNodeCoordArrayRead(UM *mesh, const Node **xy) {
    PetscErrorCode ierr;
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_WORLD,1,"node coordinates not created ... stopping\n");
    }
    ierr = VecGetArrayRead(mesh->loc,(const double **)xy); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMRestoreNodeCoordArrayRead(UM *mesh, const Node **xy) {
    PetscErrorCode ierr;
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_WORLD,1,"node coordinates not created ... stopping\n");
    }
    ierr = VecRestoreArrayRead(mesh->loc,(const double **)xy); CHKERRQ(ierr);
    return 0;
}

