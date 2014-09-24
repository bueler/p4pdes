static char help[] =
"Read in a FEM grid from PETSc binary file in parallel.\n\
Demonstrate Mat preallocation.\n\
For a coarse grid example do:\n\
     triangle -pqa1.0 bump   # generates bump.1.{node,ele}\n\
     c2triangle -f bump.1    # reads bump.1.{node,ele} and generates bump.1.petsc\n\
     c2prealloc -f bump.1    # reads bump.1.petsc\n\
To see the sparsity pattern graphically:\n\
     c2prealloc -f bump.1 -mat_view draw -draw_pause 2\n\n";

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

#define DEBUG 1

int main(int argc,char **args)
{
  // MAJOR VARIABLES
  PetscInt N,   // number of degrees of freedom (= number of all nodes)
           M;   // number of elements;
  Vec      x, y, BT, P; // mesh:  x coord of node, y coord of node, bdry type, element indexing
  Mat      A;   // we preallocate this stiffness matrix

  // INITIALIZE PETSC
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;

  // GET FILENAME FROM OPTION
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fname[MPL];
  PetscBool      fset;
  ierr = PetscOptionsBegin(COMM, "", "options for c2prealloc", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root with PETSc binary, for reading", "", "",
                            fname, sizeof(fname), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAME  required");
  }
  strcat(fname,".petsc");

  // ALLOCATE AND READ IN PARALLEL: NODE INFO
  PetscViewer viewer;
  ierr = PetscPrintf(COMM,"reading x,y,BT from %s in parallel ...\n",fname); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,
             &viewer); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&x); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&y); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&BT); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"node x coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)y,"node y coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)BT,"node boundary type"); CHKERRQ(ierr);
  ierr = VecLoad(x,viewer); CHKERRQ(ierr);
  ierr = VecLoad(y,viewer); CHKERRQ(ierr);
  ierr = VecLoad(BT,viewer); CHKERRQ(ierr);

  ierr = VecCreate(COMM,&P); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)P,"element node index array"); CHKERRQ(ierr);
  ierr = VecLoad(P,viewer); CHKERRQ(ierr);

  ierr = VecGetSize(x,&N); CHKERRQ(ierr);
  ierr = VecGetSize(P,&M); CHKERRQ(ierr);
  if (M % 3 != 0) {
    SETERRQ(COMM,3,"element node index array P invalid: must have 3 M entries");
  }
  M /= 3;
  ierr = PetscPrintf(COMM,"  N=%d nodes, M=%d elements\n",N,M); CHKERRQ(ierr);
#if DEBUG
  PetscInt bigsize=1000;
  if ((N < bigsize) && (M < bigsize)) {
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(BT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(P,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(COMM,"  [supressing VecView to STDOUT because too big]"); CHKERRQ(ierr);
  }
#endif

  // PUT A COPY OF THE FULL P ON EACH PROCESSOR
  VecScatter  ctx;
  Vec         PSEQ;
  ierr = VecScatterCreateToAll(P,&ctx,&PSEQ); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,P,PSEQ,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,P,PSEQ,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);

  // LEARN WHICH ROWS WE OWN
  PetscInt Istart,Iend;
  ierr = VecGetOwnershipRange(x,&Istart,&Iend); CHKERRQ(ierr);

  // ALLOCATE LOCAL ARRAYS FOR NUMBER OF NONZEROS
  PetscInt mm = Iend - Istart;
  int *dnnz, // dnnz[i] is number of nonzeros in row which are in same-processor column
      *onnz; // onnz[i] is number of nonzeros in row which are in other-processor column
  PetscMalloc(mm*sizeof(int),&dnnz);
  PetscMalloc(mm*sizeof(int),&onnz);

  // FILL THE NUMBER-OF-NONZEROS ARRAYS
  PetscInt    iloc, i, j, k, q, r;
  PetscScalar *ap;
  ierr = VecGetArray(PSEQ,&ap); CHKERRQ(ierr);
  for (k = 0; k < M; k++) {          // loop over ALL elements
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = ap[3*k+q] - 1;             //   ... its global node index
      // do I own it?
      if ((i >= Istart) && (i < Iend)) {
        for (r = 0; r < 3; r++) {    // loop over other vertices
          j = ap[3*k+r] - 1;         //   ... its global node index
          if ((j >= Istart) && (j < Iend))
            dnnz[i]++;
          else
            onnz[i]++;
        }
      }
    }
  }
  ierr = VecRestoreArray(PSEQ,&ap); CHKERRQ(ierr);
#if DEBUG
  PetscMPIInt rank;
  MPI_Comm_rank(COMM,&rank);
  ierr = PetscSynchronizedPrintf(COMM,"showing entries of dnnz[] on rank %d (DEBUG)\n",rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(COMM,"dnnz[%d] = %d\n",iloc,dnnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedPrintf(COMM,"showing entries of onnz[] on rank %d (DEBUG)\n",rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(COMM,"onnz[%d] = %d\n",iloc,onnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(COMM,PETSC_STDOUT); CHKERRQ(ierr);
#endif

  ierr = MatCreateAIJ(COMM,mm,mm,N,N,0,dnnz,0,onnz,&A); CHKERRQ(ierr);

//MatCreateAIJ(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
//"If the *_nnz parameter is given then the *_nz parameter is ignored"

//  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
//  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
//  MatSetFromOptions(A);

  MatDestroy(&A);

  PetscFree(dnnz);
  PetscFree(onnz);
  VecDestroy(&PSEQ);

  // CLEAN UP
  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&BT);
  VecDestroy(&P);
  PetscViewerDestroy(&viewer);

  PetscFinalize();
  return 0;
}
