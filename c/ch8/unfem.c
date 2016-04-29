
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem -un_checkload -un_mesh foo.dat

#include <petsc.h>

// should use IS for elements
//    IS e;
//    ierr = ISCreate(PETSC_COMM_WORLD,&e); CHKERRQ(ierr);
//    ierr = ISLoad(e,viewer); CHKERRQ(ierr);  <-- FAILS  see petsc issue #127
//    ierr = ISView(e,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//    ISDestroy(&e);

typedef struct _Element {
    int node[3];
} Element;

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscViewer viewer;
    Vec         xnode, ynode;
    Vec         evec;
    Element     *e;
    double      *ax, *ay, *aevec;
    int         NN, KK, k, n, m;
    PetscBool   checkload = PETSC_FALSE;
    char        meshroot[256] = "foo.dat", nodename[266], elename[266];

    PetscInitialize(&argc,&argv,NULL,help);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-checkload","check on loaded nodes and elements",
           "unfem.c",checkload,&checkload,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh","file name root of mesh (files have .node,.ele)",
           "unfem.c",meshroot,meshroot,sizeof(meshroot),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    strcpy(nodename, meshroot);
    strncat(nodename, ".node", 10);
    strcpy(elename, meshroot);
    strncat(elename, ".ele", 10);
    // ierr = PetscPrintf(PETSC_COMM_WORLD,"nodename = %s, elename = %s\n",nodename,elename); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&xnode); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&ynode); CHKERRQ(ierr);
    ierr = VecSetFromOptions(xnode); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ynode); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,nodename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(xnode,viewer); CHKERRQ(ierr);
    ierr = VecLoad(ynode,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(xnode,&NN); CHKERRQ(ierr);
    ierr = VecGetSize(ynode,&m); CHKERRQ(ierr);
    if (NN != m) {
        SETERRQ1(PETSC_COMM_WORLD,1,"xnode,ynode loaded from %s are not the same size\n",nodename);
    }

    ierr = VecCreate(PETSC_COMM_WORLD,&evec); CHKERRQ(ierr);
    ierr = VecSetFromOptions(evec); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,elename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(evec,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecGetSize(evec,&KK); CHKERRQ(ierr);
    if (KK % 3 != 0) {
        SETERRQ1(PETSC_COMM_WORLD,2,"evec loaded from %s is wrong size for list of element triples\n",elename);
    }
    KK /= 3;
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"evec is of size K = %d\n",KK); CHKERRQ(ierr);
    e = (Element*)malloc(KK * sizeof(Element));
    ierr = VecGetArray(evec,&aevec); CHKERRQ(ierr);
    for (k = 0; k < KK; k++) {
        for (m = 0; m < 3; m++)
            e[k].node[m] = aevec[3*k+m];
    }
    ierr = VecRestoreArray(evec,&aevec); CHKERRQ(ierr);
    VecDestroy(&evec);


    if (checkload) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%d nodes:\n",NN); CHKERRQ(ierr);
        ierr = VecGetArray(xnode,&ax); CHKERRQ(ierr);
        ierr = VecGetArray(ynode,&ay); CHKERRQ(ierr);
        for (n = 0; n < NN; n++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    %3d = (%g,%g)\n",
                               n,ax[n],ay[n]); CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(xnode,&ax); CHKERRQ(ierr);
        ierr = VecRestoreArray(ynode,&ay); CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD,"%d elements:\n",KK); CHKERRQ(ierr);
        for (k = 0; k < KK; k++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"    e[%3d] = %3d %3d %3d\n",
                               k,e[k].node[0],e[k].node[1],e[k].node[2]); CHKERRQ(ierr);
        }
    }

    free(e);
    VecDestroy(&xnode);  VecDestroy(&ynode);
    PetscFinalize();
    return 0;
}

