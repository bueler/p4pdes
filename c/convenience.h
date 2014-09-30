//START
#define vecassembly(X) { \
            ierr = VecAssemblyBegin(X); CHKERRQ(ierr); \
            ierr = VecAssemblyEnd(X); CHKERRQ(ierr); }
#define matassembly(X) { \
            ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); \
            ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); }
#define createloadname(X,VIEWER,NAME) { \
            ierr = VecCreate(COMM,&X); CHKERRQ(ierr); \
            ierr = VecLoad(X,VIEWER); CHKERRQ(ierr); \
            ierr = PetscObjectSetName((PetscObject)X,NAME); CHKERRQ(ierr); }
//END
