static char help[] =
"Read in a FEM grid from PETSc binary file in parallel.\n\
Demonstrate Mat preallocation.\n\n";

// do
//     triangle -pqa1.0 bump   # generates bump.1.{node,ele}
//     c2triangle -f bump.1    # reads bump.1.{node,ele} and generates bump.1.petsc
//     c2prealloc -f bump.1.petsc

// to see sparsity: ./c1prealloc -mat_view draw -draw_pause 1

// SUMMARY FROM PETSC MANUAL
/*
For (vertex-based) finite element type calculations, an analogous procedure is as follows:
  - Allocate integer array nnz.
  - Loop over vertices, computing the number of neighbor vertices, which determines the
number of nonzeros for the corresponding matrix row(s).
  - Create the sparse matrix via MatCreateSeqAIJ() or alternative.
  - Loop over elements, generating matrix entries and inserting in matrix via MatSetValues().
*/

#include <petscmat.h>
#include <petscksp.h>

#define DEBUG 0

int main(int argc,char **args)
{
  // MAJOR VARIABLES
  PetscInt N,   // number of degrees of freedom (= number of all nodes)
           M;   // number of elements;
  Vec      x, y, BT, P;
  Mat      A;   // we preallocate this stiffness matrix

  // INITIALIZE PETSC
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;

  // GET FILENAME ROOT FROM OPTION
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fname[MPL];
  PetscBool      fset;
  ierr = PetscOptionsBegin(COMM, "", "options for c2prealloc", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename with PETSc binary, for reading", "", "",
                            fname, sizeof(fname), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAME  required");
  }

  // ALLOCATE and READ IN PARALLEL
  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,
             &viewer); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&x); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&y); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&BT); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&P); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"node x coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)y,"node y coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)BT,"node boundary type"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)P,"element node index array"); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"reading x,y,BT,P from %s in parallel ...\n",fname); CHKERRQ(ierr);
  ierr = VecLoad(x,viewer); CHKERRQ(ierr);
  ierr = VecLoad(y,viewer); CHKERRQ(ierr);
  ierr = VecLoad(BT,viewer); CHKERRQ(ierr);
  ierr = VecLoad(P,viewer); CHKERRQ(ierr);

#if DEBUG
  PetscInt rN, rM, bigsize=1000;
  ierr = VecGetSize(x,&rN); CHKERRQ(ierr);
  ierr = VecGetSize(P,&rM); CHKERRQ(ierr);
  rM /= 3;
  ierr = PetscPrintf(COMM,"  N=%d nodes, M=%d elements\n",rN,rM); CHKERRQ(ierr);
  if ((rN < bigsize) && (rM < bigsize)) {
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(BT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(P,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(COMM,"  [supressing VecView to STDOUT because too big]"); CHKERRQ(ierr);
  }
#endif

FIXME FROM HERE

  // CLEAN UP
  VecDestroy(&rx);
  VecDestroy(&ry);
  VecDestroy(&rBT);
  VecDestroy(&rP);
  PetscViewerDestroy(&rviewer);

  // READ NODE HEADER
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

  // READ NODES; EACH PROCESS HOLDS FULL INFO (INDEX, LOCATION, BOUNDARY MARKER)
  PetscMalloc(N*sizeof(int),&BT);
  PetscMalloc(N*sizeof(double),&x);
  PetscMalloc(N*sizeof(double),&y);
  PetscInt i,iplusone;
  for (i = 0; i < N; i++) {
    if (4 != fscanf(nodefile,"%d %lf %lf %d\n",&iplusone,&(x[i]),&(y[i]),&(BT[i]))) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
    }
  }
#if DEBUG
  ierr = PetscPrintf(COMM,"boundary type read:\n"); CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ierr = PetscPrintf(COMM,"%4d%6d\n",i,BT[i]); CHKERRQ(ierr);
  }
#endif

  // READ ELEMENT HEADER
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

  // READ ELEMENTS; EACH PROCESS HOLDS FULL INFO (THREE NODE INDICES PER ELEMENT)
  PetscInt k, kplusone;
  PetscMalloc(M*sizeof(int*),&P);
  for (k = 0; k < M; k++) {
    PetscMalloc(3*sizeof(int),&(P[k]));
    if (4 != fscanf(elefile,"%d %d %d %d\n",&kplusone,&(P[k][0]),&(P[k][1]),&(P[k][2]))) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",elefilename);
    }
  }
#if DEBUG
  ierr = PetscPrintf(COMM,"elements read:\n"); CHKERRQ(ierr);
  for (k = 0; k < M; k++) {
    ierr = PetscPrintf(COMM,"%4d%8d%6d%6d\n",k,P[k][0],P[k][1],P[k][2]); CHKERRQ(ierr);
  }
#endif
  }

#if 0
  // LEARN WHICH ROWS WE OWN
  Vec  U;  // this Vec only exists to get ownership ranges
  PetscInt Istart,Iend;
  ierr = VecCreateMPI(COMM,PETSC_DECIDE,N,&U); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U,&Istart,&Iend); CHKERRQ(ierr);

  // ALLOCATE LOCAL ARRAYS FOR NUMBER OF NONZEROS
  PetscInt mym = Iend-Istart,
           iloc;
  int *dnnz, // dnnz[i] is number of nonzeros in row which are in same-processor column
      *onnz; // onnz[i] is number of nonzeros in row which are in other-processor column
  PetscMalloc(mym*sizeof(int*),&dnnz);
  PetscMalloc(mym*sizeof(int*),&onnz);

  // FILL THE NUMBER-OF-NONZERO ARRAYS
  for (iloc = 0; iloc < mym; iloc++) {
    dnnz[iloc] = 1;  // always have a diagonal entry
    onnz[iloc] = 0;  // zero out
  }
  PetscInt p, q, j;
  for (k = 0; k < M; k++) {          // loop over elements
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = P[k][q] - 1;               // global node index
      // do I own it?
      if ((i >= Istart) && (i < Iend)) {
        for (p = 0; p < 3; p++) {    // look at other vertices
          if (p != q) {              //   .. distinct ones
            j = P[k][p] - 1;          // its global node index
            if ((j >= Istart) && (j < Iend))
              dnnz[i]++;
            else
              onnz[i]++;
          }
        }
      }
    }
  }

#endif

#if 0
  ierr = PetscSynchronizedPrintf(COMM,"dnnz: (mym = %d)\n",mym); CHKERRQ(ierr);
  for (iloc = 0; iloc < mym; iloc++) {
    ierr = PetscSynchronizedPrintf(COMM,"%4d%6d\n",i,dnnz[i]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(COMM,PETSC_STDOUT); CHKERRQ(ierr);
#endif

#if 0
// FIXME create a AIJ Mat

  PetscInt myn = Iend-Istart; // we respect Vec ownership

//  ierr = MatCreateMPIAIJ(COMM,mym,myn,N,N,0,dnnz,0,onnz,&A); CHKERRQ(ierr);
// MatCreateMPIAIJ(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
//"If the *_nnz parameter is given then the *_nz parameter is ignored"

  ierr = MatCreate(COMM,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,mym,myn,N,N); CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ); CHKERRQ(ierr);
//  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,0,dnnz,0,onnz); CHKERRQ(ierr);

// MatMPIAIJSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])

//  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
//  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
//  MatSetFromOptions(A);

  MatDestroy(&A);

#endif

  VecDestroy(&vx);
  VecDestroy(&vy);
  VecDestroy(&vP);
  VecDestroy(&vBT);
//  PetscFree(dnnz);
//  PetscFree(onnz);

  PetscFinalize();
  return 0;
}
