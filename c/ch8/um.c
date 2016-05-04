#include <petsc.h>
#include "um.h"

PetscErrorCode UMInitialize(UM *mesh) {
    mesh->N = 0;
    mesh->K = 0;
    mesh->e = NULL;
    mesh->bf = NULL;
    mesh->x = NULL;
    mesh->y = NULL;
    return 0;
}

PetscErrorCode UMDestroy(UM *mesh) {
    ISDestroy(&(mesh->e));
    ISDestroy(&(mesh->bf));
    VecDestroy(&(mesh->x));
    VecDestroy(&(mesh->y));
    return 0;
}

PetscErrorCode UMView(UM *mesh, PetscViewer viewer) {
    PetscErrorCode ierr;
    const double *ax, *ay;
    int          n, k;
    const int    *ae, *abf;
    ierr = PetscViewerASCIIPushSynchronized(viewer); CHKERRQ(ierr);
    if ((mesh->x) && (mesh->y) && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d nodes at (x,y) coordinates:\n",mesh->N); CHKERRQ(ierr);
        ierr = VecGetArrayRead(mesh->x,&ax); CHKERRQ(ierr);
        ierr = VecGetArrayRead(mesh->y,&ay); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : (%g,%g)\n",
                               n,ax[n],ay[n]); CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(mesh->x,&ax); CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(mesh->y,&ay); CHKERRQ(ierr);
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
    if ((mesh->bf) && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary flags at nodes (0 = interior, 2 = Dirichlet):\n",mesh->N); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->bf,&abf); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %1d\n",
                               n,abf[n]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->bf,&abf); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"boundary flags empty/unallocated\n"); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopSynchronized(viewer); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMReadNodes(UM *mesh, char *rootname) {
    PetscErrorCode ierr;
    int         m;
    PetscViewer viewer;
    char        filename[266];
    strcpy(filename, rootname);
    strncat(filename, ".node", 10);
    if (mesh->N > 0) {
        SETERRQ(PETSC_COMM_WORLD,1,"nodes already created?\n");
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&mesh->x); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&mesh->y); CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->x); CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->y); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(mesh->x,viewer); CHKERRQ(ierr);
    ierr = VecLoad(mesh->y,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(mesh->x,&(mesh->N)); CHKERRQ(ierr);
    ierr = VecGetSize(mesh->y,&m); CHKERRQ(ierr);
    if (mesh->N != m) {
        SETERRQ1(PETSC_COMM_WORLD,2,"node coordinates x,y loaded from %s are not the same size\n",filename);
    }
    return 0;
}

PetscErrorCode UMReadElements(UM *mesh, char *rootname) {
    PetscErrorCode ierr;
    PetscViewer viewer;
    int         n_bf;
    char        filename[266];
    strcpy(filename, rootname);
    strncat(filename, ".ele", 10);
    if (mesh->K > 0) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "elements already created? ... stopping\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "node coordinates not created ... do that first ... stopping\n");
    }
    // create and load e
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->e)); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = ISLoad(mesh->e,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->e,&(mesh->K)); CHKERRQ(ierr);
    if (mesh->K % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,3,
                 "IS e loaded from %s is wrong size for list of element triples\n",filename);
    }
    mesh->K /= 3;
    // create and load bf
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->bf)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->bf,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->bf,&n_bf); CHKERRQ(ierr);
    if (n_bf != mesh->N) {
        SETERRQ1(PETSC_COMM_WORLD,4,
                 "IS bf loaded from %s is wrong size\n",filename);
    }
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
        // FIXME: could add check of distinct indices
    }
    ierr = ISRestoreIndices(mesh->e,&ae); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMCheckBoundaryFlags(UM *mesh) {
    PetscErrorCode ierr;
    const int   *abf;
    int         n;
    if (mesh->bf == NULL) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "boundary flags not allocated; call UMReadNodes() first\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_WORLD,2,
                "node size unknown so boundary flag check impossible; call UMReadNodes() first\n");
    }
    ierr = ISGetIndices(mesh->bf,&abf); CHKERRQ(ierr);
    for (n = 0; n < mesh->N; n++) {
        switch (abf[n]) {
            case 0 :
            case 1 :
            case 2 :
                break;
            default :
                SETERRQ2(PETSC_COMM_WORLD,3,
                   "boundary flag bf[%d]=%d invalid: not in {0,1,2}\n",
                   n,abf[n]);
        }
    }
    ierr = ISRestoreIndices(mesh->bf,&abf); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMAssertValid(UM *mesh) {
    PetscErrorCode ierr;
    if ((mesh->N == 0) || (!(mesh->x)) || (!(mesh->y))) {
        SETERRQ(PETSC_COMM_WORLD,1,
                "nodes not created; call UMReadNodes() first\n");
    }
    ierr = UMCheckElements(mesh); CHKERRQ(ierr);
    ierr = UMCheckBoundaryFlags(mesh); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMCreateGlobalVec(UM *mesh, Vec *v) {
    PetscErrorCode ierr;
    ierr = UMAssertValid(mesh); CHKERRQ(ierr);
    ierr = VecDuplicate(mesh->x,v); CHKERRQ(ierr);
    return 0;
}

