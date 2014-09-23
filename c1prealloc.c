static char help[] = "Read in a FEM grid.  Demonstrate Mat preallocation.\n\n";

// to see sparsity: ./c1prealloc -mat_view draw -draw_pause 1

#include <petscmat.h>
#include <petscksp.h>

int main(int argc,char **args)
{
  PetscInt N,   // number of degrees of freedom (= number of all nodes)
           M;   // number of elements;
  int **P,      // array with M rows and 3 columns; P[k][q] is node index
      *BT;      // array with N rows and 2 columns;
                //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
  int *nnz;     // nnz[i] is number of nonzeros in row

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm COMM = PETSC_COMM_WORLD;
  PetscErrorCode ierr;

  // do   triangle -pqa1.0 bump  (or similar) to generate bump.1.{node,ele}

  // FIXME use args for fnameroot
  FILE           *nodefile, *elefile;
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fnameroot[MPL], nodefilename[MPL], elefilename[MPL];
  strcpy(fnameroot,"bump.1");
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");

  PetscInt ndim, nattr, nbdrymarkers;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  if (4 != fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers)) {
    SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
  }
  if (ndim != 2) {
    SETERRQ1(COMM,1,"ndim read from %s not equal to 2",nodefilename);
  }
  ierr = PetscPrintf(COMM,"read %s:\n",nodefilename); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "  N=%d nodes in 2D polygon with %d attributes and %d boundary markers per node\n",
           N,nattr,nbdrymarkers); CHKERRQ(ierr);
  PetscMalloc(N*sizeof(int),&BT);
  PetscInt i,iplusone;
  double x,y;
  for (i = 0; i < N; i++) {
    if (4 != fscanf(nodefile,"%d %lf %lf %d\n",&iplusone,&x,&y,&(BT[i]))) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
    }
  }
  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
#if 0
  ierr = PetscPrintf(COMM,"boundary type read:\n"); CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = PetscPrintf(COMM,"%4d%6d\n",i,BT[i]); CHKERRQ(ierr);
  }
#endif

  PetscInt nthree, nattrele;
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);
  if (3 != fscanf(elefile,"%d %d %d\n",&M,&nthree,&nattrele)) {
    SETERRQ1(COMM,1,"expected 3 values in reading from %s",elefilename);
  }
  if (nthree != 3) {
    SETERRQ1(COMM,1,"nthree read from %s not equal to 3 (= nodes per element)",elefilename);
  }
  ierr = PetscPrintf(COMM,"read %s:\n",elefilename); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "  M=%d elements in 2D polygon with %d attributes per element\n",
           M,nattrele); CHKERRQ(ierr);
  
  PetscInt k, kplusone;
  PetscMalloc(M*sizeof(int*),&P);
  for (k = 0; k < M; k++) {
    PetscMalloc(3*sizeof(int),&(P[k]));
    if (4 != fscanf(elefile,"%d %d %d %d\n",&kplusone,&(P[k][0]),&(P[k][1]),&(P[k][2]))) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",elefilename);
    }
  }
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);
#if 0
  ierr = PetscPrintf(COMM,"elements read:\n"); CHKERRQ(ierr);
  for (k = 0; k < M; k++) {
    ierr = PetscPrintf(COMM,"%4d%8d%6d%6d\n",k,P[k][0],P[k][1],P[k][2]); CHKERRQ(ierr);
  }
#endif

  PetscMalloc(N*sizeof(int*),&nnz);
  for (i = 0; i < N; i++) {
    nnz[i] = 1;  // always a diagonal entry
  }

  PetscInt q;
  for (k = 0; k < M; k++) {
    for (q = 0; q < 3; q++) {
      i = P[k][q] - 1;
      if (BT[i] != 2)
        nnz[i]++;
    }
  }
#if 0
  ierr = PetscPrintf(COMM,"nnz:\n"); CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = PetscPrintf(COMM,"%4d%6d\n",i,nnz[i]); CHKERRQ(ierr);
  }
#endif

// FIXME actually read the info so as to preallocate

  PetscFinalize();
  return 0;
}
