#include <petsc.h>
#include "um.h"

PetscErrorCode UMInitialize(UM *mesh) {
    mesh->N = 0;
    mesh->K = 0;
    mesh->P = 0;
    mesh->loc = NULL;
    mesh->e = NULL;
    mesh->bf = NULL;
    mesh->ns = NULL;
    return 0;
}

PetscErrorCode UMDestroy(UM *mesh) {
    PetscErrorCode ierr;
    ierr = VecDestroy(&(mesh->loc)); CHKERRQ(ierr);
    ierr = ISDestroy(&(mesh->e)); CHKERRQ(ierr);
    ierr = ISDestroy(&(mesh->bf)); CHKERRQ(ierr);
    ierr = ISDestroy(&(mesh->ns)); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode UMViewASCII(UM *mesh, PetscViewer viewer) {
    PetscErrorCode ierr;
    PetscInt        n, k;
    const Node      *aloc;
    const PetscInt  *ae, *abf, *ans;

    ierr = PetscViewerASCIIPushSynchronized(viewer); CHKERRQ(ierr);
    if (mesh->loc && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d nodes at (x,y) coordinates:\n",mesh->N); CHKERRQ(ierr);
        ierr = VecGetArrayRead(mesh->loc,(const PetscReal **)&aloc); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : (%g,%g)\n",
                               n,aloc[n].x,aloc[n].y); CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(mesh->loc,(const PetscReal **)&aloc); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"node coordinates empty or unallocated\n"); CHKERRQ(ierr);
    }
    if (mesh->e && (mesh->K > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d elements:\n",mesh->K); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->e,&ae); CHKERRQ(ierr);
        for (k = 0; k < mesh->K; k++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d : %3d %3d %3d\n",
                               k,ae[3*k+0],ae[3*k+1],ae[3*k+2]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->e,&ae); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"element index triples empty or unallocated\n"); CHKERRQ(ierr);
    }
    if (mesh->bf && (mesh->N > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary flags at nodes (0 = interior, 1 = boundary, 2 = Dirichlet):\n",mesh->N); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->bf,&abf); CHKERRQ(ierr);
        for (n = 0; n < mesh->N; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %1d\n",
                               n,abf[n]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->bf,&abf); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"boundary flags empty or unallocated\n"); CHKERRQ(ierr);
    }
    if (mesh->ns && (mesh->P > 0)) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d Neumann boundary segments:\n",mesh->P); CHKERRQ(ierr);
        ierr = ISGetIndices(mesh->ns,&ans); CHKERRQ(ierr);
        for (n = 0; n < mesh->P; n++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %3d %3d\n",
                               n,ans[2*n+0],ans[2*n+1]); CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(mesh->ns,&ans); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Neumann boundary segments empty or unallocated\n"); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopSynchronized(viewer); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMViewSolutionBinary(UM *mesh, char *filename, Vec u) {
    PetscErrorCode ierr;
    PetscInt       Nu;
    PetscViewer viewer;
    ierr = VecGetSize(u,&Nu); CHKERRQ(ierr);
    if (Nu != mesh->N) {
        SETERRQ2(PETSC_COMM_SELF,1,
           "incompatible sizes of u (=%d) and number of nodes (=%d)\n",Nu,mesh->N);
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    ierr = VecView(u,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMReadNodes(UM *mesh, char *filename) {
    PetscErrorCode ierr;
    PetscInt       twoN;
    PetscViewer viewer;
    if (mesh->N > 0) {
        SETERRQ(PETSC_COMM_SELF,1,"nodes already created?\n");
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&mesh->loc); CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->loc); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(mesh->loc,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(mesh->loc,&twoN); CHKERRQ(ierr);
    if (twoN % 2 != 0) {
        SETERRQ1(PETSC_COMM_SELF,2,"node locations loaded from %s are not N pairs\n",filename);
    }
    mesh->N = twoN / 2;
    return 0;
}


PetscErrorCode UMCheckElements(UM *mesh) {
    PetscErrorCode ierr;
    const PetscInt  *ae;
    PetscInt        k, m;
    if ((mesh->K == 0) || (mesh->e == NULL)) {
        SETERRQ(PETSC_COMM_SELF,1,
                "number of elements unknown; call UMReadElements() first\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_SELF,2,
                "node size unknown so element check impossible; call UMReadNodes() first\n");
    }
    ierr = ISGetIndices(mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < mesh->K; k++) {
        for (m = 0; m < 3; m++) {
            if ((ae[3*k+m] < 0) || (ae[3*k+m] >= mesh->N)) {
                SETERRQ3(PETSC_COMM_SELF,3,
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
    const PetscInt  *ans, *abf;
    PetscInt        n, m;
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_SELF,2,
                "node size unknown so boundary flag check impossible; call UMReadNodes() first\n");
    }
    if (mesh->bf == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,
                "boundary flags at nodes not allocated; call UMReadNodes() first\n");
    }
    if ((mesh->P > 0) && (mesh->ns == NULL)) {
        SETERRQ(PETSC_COMM_SELF,3,
                "inconsistent data for Neumann boundary segments\n");
    }
    ierr = ISGetIndices(mesh->bf,&abf); CHKERRQ(ierr);
    for (n = 0; n < mesh->N; n++) {
        switch (abf[n]) {
            case 0 :
            case 1 :
            case 2 :
                break;
            default :
                SETERRQ2(PETSC_COMM_SELF,5,
                   "boundary flag bf[%d]=%d invalid: not in {0,1,2}\n",
                   n,abf[n]);
        }
    }
    ierr = ISRestoreIndices(mesh->bf,&abf); CHKERRQ(ierr);
    if (mesh->P > 0) {
        ierr = ISGetIndices(mesh->ns,&ans); CHKERRQ(ierr);
        for (n = 0; n < mesh->P; n++) {
            for (m = 0; m < 2; m++) {
                if ((ans[2*n+m] < 0) || (ans[2*n+m] >= mesh->N)) {
                    SETERRQ3(PETSC_COMM_SELF,6,
                       "index ns[%d]=%d invalid: not between 0 and N-1=%d\n",
                       2*n+m,ans[3*n+m],mesh->N-1);
                }
            }
        }
        ierr = ISRestoreIndices(mesh->ns,&ans); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode UMReadISs(UM *mesh, char *filename) {
    PetscErrorCode ierr;
    PetscViewer  viewer;
    PetscInt     n_bf;
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_SELF,2,
                "node coordinates not created ... do that first ... stopping\n");
    }
    if ((mesh->K > 0) || (mesh->P > 0) || (mesh->e != NULL) || (mesh->bf != NULL) || (mesh->ns != NULL)) {
        SETERRQ(PETSC_COMM_SELF,1,
                "elements, boundary flags, Neumann boundary segments already created? ... stopping\n");
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    // create and load e
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->e)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->e,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->e,&(mesh->K)); CHKERRQ(ierr);
    if (mesh->K % 3 != 0) {
        SETERRQ1(PETSC_COMM_SELF,3,
                 "IS e loaded from %s is wrong size for list of element triples\n",filename);
    }
    mesh->K /= 3;
    // create and load bf
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->bf)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->bf,viewer); CHKERRQ(ierr);
    ierr = ISGetSize(mesh->bf,&n_bf); CHKERRQ(ierr);
    if (n_bf != mesh->N) {
        SETERRQ1(PETSC_COMM_SELF,4,
                 "IS bf loaded from %s is wrong size for list of boundary flags\n",filename);
    }
    // FIXME  seems there is no way to tell if file is empty at this point
    // create and load ns last ... may *start with a negative value* in which case set P = 0
    const PetscInt *ans;
    ierr = ISCreate(PETSC_COMM_WORLD,&(mesh->ns)); CHKERRQ(ierr);
    ierr = ISLoad(mesh->ns,viewer); CHKERRQ(ierr);
    ierr = ISGetIndices(mesh->ns,&ans); CHKERRQ(ierr);
    if (ans[0] < 0) {
        ISDestroy(&(mesh->ns));
        mesh->ns = NULL;
        mesh->P = 0;
    } else {
        ierr = ISGetSize(mesh->ns,&(mesh->P)); CHKERRQ(ierr);
        if (mesh->P % 2 != 0) {
            SETERRQ1(PETSC_COMM_SELF,4,
                     "IS s loaded from %s is wrong size for list of Neumann boundary segment pairs\n",filename);
        }
        mesh->P /= 2;
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    // check that mesh is complete now
    ierr = UMCheckElements(mesh); CHKERRQ(ierr);
    ierr = UMCheckBoundaryData(mesh); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMStats(UM *mesh, PetscReal *maxh, PetscReal *meanh,
                       PetscReal *maxa, PetscReal *meana) {
    PetscErrorCode ierr;
    const PetscInt *ae;
    const Node     *aloc;
    PetscInt       k;
    PetscReal      x[3], y[3], ax, ay, bx, by, cx, cy, h, a,
                   Maxh = 0.0, Maxa = 0.0, Sumh = 0.0, Suma = 0.0;
    if ((mesh->K == 0) || (mesh->e == NULL)) {
        SETERRQ(PETSC_COMM_SELF,1,
                "number of elements unknown; call UMReadElements() first\n");
    }
    if (mesh->N == 0) {
        SETERRQ(PETSC_COMM_SELF,2,
                "node size unknown so element check impossible; call UMReadNodes() first\n");
    }
    ierr = UMGetNodeCoordArrayRead(mesh,&aloc); CHKERRQ(ierr);
    ierr = ISGetIndices(mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < mesh->K; k++) {
        x[0] = aloc[ae[3*k]].x;
        y[0] = aloc[ae[3*k]].y;
        x[1] = aloc[ae[3*k+1]].x;
        y[1] = aloc[ae[3*k+1]].y;
        x[2] = aloc[ae[3*k+2]].x;
        y[2] = aloc[ae[3*k+2]].y;
        ax = x[1] - x[0];
        ay = y[1] - y[0];
        bx = x[2] - x[0];
        by = y[2] - y[0];
        cx = x[1] - x[2];
        cy = y[1] - y[2];
        h = PetscMax(ax*ax+ay*ay, PetscMax(bx*bx+by*by, cx*cx+cy*cy));
        h = sqrt(h);
        a = 0.5 * PetscAbs(ax*by-ay*bx);
        Maxh = PetscMax(Maxh,h);
        Sumh += h;
        Maxa = PetscMax(Maxa,a);
        Suma += a;
    }
    ierr = ISRestoreIndices(mesh->e,&ae); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(mesh,&aloc); CHKERRQ(ierr);
    if (maxh)  *maxh = Maxh;
    if (maxa)  *maxa = Maxa;
    if (meanh)  *meanh = Sumh / mesh->K;
    if (meana)  *meana = Suma / mesh->K;
    return 0;
}

PetscErrorCode UMGetNodeCoordArrayRead(UM *mesh, const Node **xy) {
    PetscErrorCode ierr;
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_SELF,1,"node coordinates not created ... stopping\n");
    }
    ierr = VecGetArrayRead(mesh->loc,(const PetscReal **)xy); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode UMRestoreNodeCoordArrayRead(UM *mesh, const Node **xy) {
    PetscErrorCode ierr;
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_SELF,1,"node coordinates not created ... stopping\n");
    }
    ierr = VecRestoreArrayRead(mesh->loc,(const PetscReal **)xy); CHKERRQ(ierr);
    return 0;
}

