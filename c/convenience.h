//START
#define vecassembly(X) { \
            ierr = VecAssemblyBegin(X); CHKERRQ(ierr); \
            ierr = VecAssemblyEnd(X); CHKERRQ(ierr); }
#define matassembly(X) { \
            ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); \
            ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); }
//END
