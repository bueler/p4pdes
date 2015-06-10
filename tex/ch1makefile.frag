//START
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

c1e: c1e.o  chkopts
    -${CLINKER} -o c1e c1e.o  ${PETSC_LIB}
    ${RM} c1e.o
//STOP

