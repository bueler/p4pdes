include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

e: e.o chkopts
    -${CLINKER} -o e e.o  ${PETSC_LIB}
    ${RM} e.o
