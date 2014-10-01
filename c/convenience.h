//START
#define vecassembly(X) { \
    ierr = VecAssemblyBegin(X); CHKERRQ(ierr); \
    ierr = VecAssemblyEnd(X); CHKERRQ(ierr); }
#define matassembly(X) { \
    ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); \
    ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); }
#define scatterforwardall(CTX,X,XSEQ) { \
    ierr = VecScatterCreateToAll(X,&CTX,&XSEQ); CHKERRQ(ierr); \
    ierr = VecScatterBegin(CTX,X,XSEQ,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr); \
    ierr = VecScatterEnd(CTX,X,XSEQ,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr); \
    VecScatterDestroy(&ctx); }
//END
