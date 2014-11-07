//START
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

c1e: c1e.o  chkopts
    -${CLINKER} -o c1e c1e.o  ${PETSC_LIB}
    ${RM} c1e.o
//STOP

