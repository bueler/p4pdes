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
    PetscCall(VecDestroy(&(mesh->loc)));
    PetscCall(ISDestroy(&(mesh->e)));
    PetscCall(ISDestroy(&(mesh->bf)));
    PetscCall(ISDestroy(&(mesh->ns)));
    return 0;
}

PetscErrorCode UMViewASCII(UM *mesh, PetscViewer viewer) {
    PetscInt        n, k;
    const Node      *aloc;
    const PetscInt  *ae, *abf, *ans;

    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    if (mesh->loc && (mesh->N > 0)) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%d nodes at (x,y) coordinates:\n",mesh->N));
        PetscCall(VecGetArrayRead(mesh->loc,(const PetscReal **)&aloc));
        for (n = 0; n < mesh->N; n++) {
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : (%g,%g)\n",
                               n,aloc[n].x,aloc[n].y));
        }
        PetscCall(VecRestoreArrayRead(mesh->loc,(const PetscReal **)&aloc));
    } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"node coordinates empty or unallocated\n"));
    }
    if (mesh->e && (mesh->K > 0)) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%d elements:\n",mesh->K));
        PetscCall(ISGetIndices(mesh->e,&ae));
        for (k = 0; k < mesh->K; k++) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    %3d : %3d %3d %3d\n",
                               k,ae[3*k+0],ae[3*k+1],ae[3*k+2]));
        }
        PetscCall(ISRestoreIndices(mesh->e,&ae));
    } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"element index triples empty or unallocated\n"));
    }
    if (mesh->bf && (mesh->N > 0)) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%d boundary flags at nodes (0 = interior, 1 = boundary, 2 = Dirichlet):\n",mesh->N));
        PetscCall(ISGetIndices(mesh->bf,&abf));
        for (n = 0; n < mesh->N; n++) {
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %1d\n",
                               n,abf[n]));
        }
        PetscCall(ISRestoreIndices(mesh->bf,&abf));
    } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"boundary flags empty or unallocated\n"));
    }
    if (mesh->ns && (mesh->P > 0)) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%d Neumann boundary segments:\n",mesh->P));
        PetscCall(ISGetIndices(mesh->ns,&ans));
        for (n = 0; n < mesh->P; n++) {
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    %3d : %3d %3d\n",
                               n,ans[2*n+0],ans[2*n+1]));
        }
        PetscCall(ISRestoreIndices(mesh->ns,&ans));
    } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Neumann boundary segments empty or unallocated\n"));
    }
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    return 0;
}


PetscErrorCode UMViewSolutionBinary(UM *mesh, char *filename, Vec u) {
    PetscInt       Nu;
    PetscViewer viewer;
    PetscCall(VecGetSize(u,&Nu));
    if (Nu != mesh->N) {
        SETERRQ(PETSC_COMM_SELF,1,
           "incompatible sizes of u (=%d) and number of nodes (=%d)\n",Nu,mesh->N);
    }
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(u,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    return 0;
}


PetscErrorCode UMReadNodes(UM *mesh, char *filename) {
    PetscInt       twoN;
    PetscViewer viewer;
    if (mesh->N > 0) {
        SETERRQ(PETSC_COMM_SELF,1,"nodes already created?\n");
    }
    PetscCall(VecCreate(PETSC_COMM_WORLD,&mesh->loc));
    PetscCall(VecSetFromOptions(mesh->loc));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(VecLoad(mesh->loc,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecGetSize(mesh->loc,&twoN));
    if (twoN % 2 != 0) {
        SETERRQ(PETSC_COMM_SELF,2,"node locations loaded from %s are not N pairs\n",filename);
    }
    mesh->N = twoN / 2;
    return 0;
}


PetscErrorCode UMCheckElements(UM *mesh) {
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
    PetscCall(ISGetIndices(mesh->e,&ae));
    for (k = 0; k < mesh->K; k++) {
        for (m = 0; m < 3; m++) {
            if ((ae[3*k+m] < 0) || (ae[3*k+m] >= mesh->N)) {
                SETERRQ(PETSC_COMM_SELF,3,
                   "index e[%d]=%d invalid: not between 0 and N-1=%d\n",
                   3*k+m,ae[3*k+m],mesh->N-1);
            }
        }
        // FIXME: could add check for distinct indices
    }
    PetscCall(ISRestoreIndices(mesh->e,&ae));
    return 0;
}

PetscErrorCode UMCheckBoundaryData(UM *mesh) {
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
    PetscCall(ISGetIndices(mesh->bf,&abf));
    for (n = 0; n < mesh->N; n++) {
        switch (abf[n]) {
            case 0 :
            case 1 :
            case 2 :
                break;
            default :
                SETERRQ(PETSC_COMM_SELF,5,
                   "boundary flag bf[%d]=%d invalid: not in {0,1,2}\n",
                   n,abf[n]);
        }
    }
    PetscCall(ISRestoreIndices(mesh->bf,&abf));
    if (mesh->P > 0) {
        PetscCall(ISGetIndices(mesh->ns,&ans));
        for (n = 0; n < mesh->P; n++) {
            for (m = 0; m < 2; m++) {
                if ((ans[2*n+m] < 0) || (ans[2*n+m] >= mesh->N)) {
                    SETERRQ(PETSC_COMM_SELF,6,
                       "index ns[%d]=%d invalid: not between 0 and N-1=%d\n",
                       2*n+m,ans[3*n+m],mesh->N-1);
                }
            }
        }
        PetscCall(ISRestoreIndices(mesh->ns,&ans));
    }
    return 0;
}

PetscErrorCode UMReadISs(UM *mesh, char *filename) {
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
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    // create and load e
    PetscCall(ISCreate(PETSC_COMM_WORLD,&(mesh->e)));
    PetscCall(ISLoad(mesh->e,viewer));
    PetscCall(ISGetSize(mesh->e,&(mesh->K)));
    if (mesh->K % 3 != 0) {
        SETERRQ(PETSC_COMM_SELF,3,
                 "IS e loaded from %s is wrong size for list of element triples\n",filename);
    }
    mesh->K /= 3;
    // create and load bf
    PetscCall(ISCreate(PETSC_COMM_WORLD,&(mesh->bf)));
    PetscCall(ISLoad(mesh->bf,viewer));
    PetscCall(ISGetSize(mesh->bf,&n_bf));
    if (n_bf != mesh->N) {
        SETERRQ(PETSC_COMM_SELF,4,
                 "IS bf loaded from %s is wrong size for list of boundary flags\n",filename);
    }
    // FIXME  seems there is no way to tell if file is empty at this point
    // create and load ns last ... may *start with a negative value* in which case set P = 0
    const PetscInt *ans;
    PetscCall(ISCreate(PETSC_COMM_WORLD,&(mesh->ns)));
    PetscCall(ISLoad(mesh->ns,viewer));
    PetscCall(ISGetIndices(mesh->ns,&ans));
    if (ans[0] < 0) {
        ISDestroy(&(mesh->ns));
        mesh->ns = NULL;
        mesh->P = 0;
    } else {
        PetscCall(ISGetSize(mesh->ns,&(mesh->P)));
        if (mesh->P % 2 != 0) {
            SETERRQ(PETSC_COMM_SELF,4,
                     "IS s loaded from %s is wrong size for list of Neumann boundary segment pairs\n",filename);
        }
        mesh->P /= 2;
    }
    PetscCall(PetscViewerDestroy(&viewer));

    // check that mesh is complete now
    PetscCall(UMCheckElements(mesh));
    PetscCall(UMCheckBoundaryData(mesh));
    return 0;
}


PetscErrorCode UMStats(UM *mesh, PetscReal *maxh, PetscReal *meanh,
                       PetscReal *maxa, PetscReal *meana) {
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
    PetscCall(UMGetNodeCoordArrayRead(mesh,&aloc));
    PetscCall(ISGetIndices(mesh->e,&ae));
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
    PetscCall(ISRestoreIndices(mesh->e,&ae));
    PetscCall(UMRestoreNodeCoordArrayRead(mesh,&aloc));
    if (maxh)  *maxh = Maxh;
    if (maxa)  *maxa = Maxa;
    if (meanh)  *meanh = Sumh / mesh->K;
    if (meana)  *meana = Suma / mesh->K;
    return 0;
}

PetscErrorCode UMGetNodeCoordArrayRead(UM *mesh, const Node **xy) {
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_SELF,1,"node coordinates not created ... stopping\n");
    }
    PetscCall(VecGetArrayRead(mesh->loc,(const PetscReal **)xy));
    return 0;
}


PetscErrorCode UMRestoreNodeCoordArrayRead(UM *mesh, const Node **xy) {
    if ((!mesh->loc) || (mesh->N == 0)) {
        SETERRQ(PETSC_COMM_SELF,1,"node coordinates not created ... stopping\n");
    }
    PetscCall(VecRestoreArrayRead(mesh->loc,(const PetscReal **)xy));
    return 0;
}
