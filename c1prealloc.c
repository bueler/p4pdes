static char help[] = "Read in a FEM grid.  Demonstrate Mat preallocation.\n\n";

// do   triangle -pqa1.0 bump  (or similar) to generate bump.1.{node,ele}
// then do
//    ./c1prealloc -f bump.1
// to see sparsity: ./c1prealloc -mat_view draw -draw_pause 1

#include <petscmat.h>
#include <petscksp.h>

#define DEBUG 0

int main(int argc,char **args)
{
  // MAJOR VARIABLES
  PetscInt N,   // number of degrees of freedom (= number of all nodes)
           M;   // number of elements;
  double *x,*y; // (x[i],y[i]) is location of node
  int **P,      // array with M rows and 3 columns; P[k][q] is node index
      *BT;      // array with N rows and 2 columns;
                //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
  int *nnz;     // nnz[i] is number of nonzeros in row

  // INITIALIZE PETSC
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm COMM = PETSC_COMM_WORLD;
  PetscErrorCode ierr;

  // GET FILENAME ROOT FROM OPTION
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fnameroot[MPL];
  PetscBool      fset;
  ierr = PetscOptionsBegin(COMM, "", "options for c1prealloc", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "triangle filename root to read", "", "bump.1",
                            fnameroot, sizeof(fnameroot), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAMEROOT  required");
  }

  // DETERMINE FILENAMES AND OPEN FILES
  FILE           *nodefile, *elefile;
  char           nodefilename[MPL], elefilename[MPL];
  strcpy(fnameroot,"bump.1");
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);

  // READ NODES: INDEX, LOCATION, BOUNDARY MARKER
  PetscInt ndim, nattr, nbdrymarkers;
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
  PetscMalloc(N*sizeof(double),&x);
  PetscMalloc(N*sizeof(double),&y);
  PetscInt i,iplusone;
  for (i = 0; i < N; i++) {
    if (4 != fscanf(nodefile,"%d %lf %lf %d\n",&iplusone,&(x[i]),&(y[i]),&(BT[i]))) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
    }
  }
  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
#if DEBUG
  ierr = PetscPrintf(COMM,"boundary type read:\n"); CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = PetscPrintf(COMM,"%4d%6d\n",i,BT[i]); CHKERRQ(ierr);
  }
#endif

  // READ ELEMENTS: THREE NODE INDICES PER ELEMENT
  PetscInt nthree, nattrele;
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
#if DEBUG
  ierr = PetscPrintf(COMM,"elements read:\n"); CHKERRQ(ierr);
  for (k = 0; k < M; k++) {
    ierr = PetscPrintf(COMM,"%4d%8d%6d%6d\n",k,P[k][0],P[k][1],P[k][2]); CHKERRQ(ierr);
  }
#endif

  // BUILD ARRAY FOR NUMBER OF NONZEROS
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
#if DEBUG
  ierr = PetscPrintf(COMM,"nnz:\n"); CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = PetscPrintf(COMM,"%4d%6d\n",i,nnz[i]); CHKERRQ(ierr);
  }
#endif

// FIXME actually read the info so as to preallocate

  PetscFinalize();
  return 0;
}
