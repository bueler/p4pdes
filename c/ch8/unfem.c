
static char help[] = "Test load of Vecs and IS.\n\n";

// example:
//   $ ./tri2petsc.py meshes/blob.1 foo.dat
//   $ ./unfem

#include <petsc.h>

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscViewer viewer;
    Vec         xnode, ynode;
    // for IS version see petsc issue #127
    //IS          e;
    Vec         e;

    PetscInitialize(&argc,&argv,NULL,help);

    ierr = VecCreate(PETSC_COMM_WORLD,&xnode); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&ynode); CHKERRQ(ierr);

    //ierr = ISCreate(PETSC_COMM_WORLD,&e); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&e); CHKERRQ(ierr);

    ierr = VecSetFromOptions(xnode); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ynode); CHKERRQ(ierr);
    ierr = VecSetFromOptions(e); CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"foo.dat.node",FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(xnode,viewer); CHKERRQ(ierr);
    ierr = VecLoad(ynode,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"foo.dat.ele",FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    //ierr = ISLoad(e,viewer); CHKERRQ(ierr);
    ierr = VecLoad(e,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = VecView(xnode,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(ynode,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    //ierr = ISView(e,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(e,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    VecDestroy(&xnode);  VecDestroy(&ynode);
    //ISDestroy(&e);
    VecDestroy(&e);
    PetscFinalize();
    return 0;
}

