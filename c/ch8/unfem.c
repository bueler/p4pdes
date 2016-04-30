
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_check -un_mesh foo.dat

#include <petsc.h>


typedef struct {
    int node[3];
} Element;

typedef struct {
    int      N,     // number of nodes
             K;     // number of elements
    Vec      x, y;  // coordinates of nodes
    Element  *e;    // in
} UnCtx;

//FIXME add Viewer argument?
PetscErrorCode UnCtxView(UnCtx *ctx) {
    PetscErrorCode ierr;
    double  *ax, *ay;
    int     n, k;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d nodes:\n",ctx->N); CHKERRQ(ierr);
    ierr = VecGetArray(ctx->x,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(ctx->y,&ay); CHKERRQ(ierr);
    for (n = 0; n < ctx->N; n++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d = (%g,%g)\n",
                           n,ax[n],ay[n]); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(ctx->x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(ctx->y,&ay); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d elements:\n",ctx->K); CHKERRQ(ierr);
    for (k = 0; k < ctx->K; k++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    e[%3d] = %3d %3d %3d\n",
                           k,ctx->e[k].node[0],ctx->e[k].node[1],ctx->e[k].node[2]); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode UnCtxReadNodes(UnCtx *ctx, char *filename) {
    PetscErrorCode ierr;
    int         m;
    PetscViewer viewer;
    ierr = VecCreate(PETSC_COMM_WORLD,&ctx->x); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&ctx->y); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ctx->x); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ctx->y); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(ctx->x,viewer); CHKERRQ(ierr);
    ierr = VecLoad(ctx->y,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(ctx->x,&(ctx->N)); CHKERRQ(ierr);
    ierr = VecGetSize(ctx->y,&m); CHKERRQ(ierr);
    if (ctx->N != m) {
        SETERRQ1(PETSC_COMM_WORLD,1,"node coordinates x,y loaded from %s are not the same size\n",filename);
    }
    return 0;
}

//FIXME should use IS for elements
//    IS e;
//    ierr = ISCreate(PETSC_COMM_WORLD,&e); CHKERRQ(ierr);
//    ierr = ISLoad(e,viewer); CHKERRQ(ierr);  <-- FAILS  see petsc issue #127
//    ierr = ISView(e,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//    ISDestroy(&e);

PetscErrorCode UnCtxReadElements(UnCtx *ctx, char *filename) {
    PetscErrorCode ierr;
    double      *ae;
    int         k, m;
    Vec         evec;
    PetscViewer viewer;
    ierr = VecCreate(PETSC_COMM_WORLD,&evec); CHKERRQ(ierr);
    ierr = VecSetFromOptions(evec); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(evec,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(evec,&k); CHKERRQ(ierr);
    if (k % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"evec loaded from %s is wrong size for list of element triples\n",filename);
    }
    ctx->K = k/3;
    if (ctx->e) {
        SETERRQ(PETSC_COMM_WORLD,2,"something wrong ... ctx->e already malloced\n");
    }
    ctx->e = (Element*)malloc(ctx->K * sizeof(Element));
    ierr = VecGetArray(evec,&ae); CHKERRQ(ierr);
    for (k = 0; k < ctx->K; k++) {
        for (m = 0; m < 3; m++)
            ctx->e[k].node[m] = ae[3*k+m];
    }
    ierr = VecRestoreArray(evec,&ae); CHKERRQ(ierr);
    VecDestroy(&evec);
    return 0;
}

PetscErrorCode UnCtxDestroy(UnCtx *ctx) {
    if (ctx->e)
        free(ctx->e);
    VecDestroy(&(ctx->x));
    VecDestroy(&(ctx->y));
    return 0;
}


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscBool   check = PETSC_FALSE;
    char        meshroot[256] = "", nodename[266], elename[266];
    UnCtx       mesh;

    PetscInitialize(&argc,&argv,NULL,help);
    mesh.e = NULL;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-check",
           "check on loaded nodes and elements",
           "unfem.c",check,&check,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh (files have .node,.ele extensions)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    strcpy(nodename, meshroot);
    strncat(nodename, ".node", 10);
    strcpy(elename, meshroot);
    strncat(elename, ".ele", 10);

    ierr = UnCtxReadNodes(&mesh,nodename); CHKERRQ(ierr);
    ierr = UnCtxReadElements(&mesh,elename); CHKERRQ(ierr);
    if (check) {
        ierr = UnCtxView(&mesh); CHKERRQ(ierr);
    }

    UnCtxDestroy(&mesh);
    PetscFinalize();
    return 0;
}

