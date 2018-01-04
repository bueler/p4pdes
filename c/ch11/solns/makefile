include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

dam: dam.o ../../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o dam dam.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} dam.o ../../ch6/poissonfunctions.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ dam

