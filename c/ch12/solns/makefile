include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

dam: dam.o ../../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o dam dam.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} dam.o ../../ch6/poissonfunctions.o

elasto: elasto.o ../../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o elasto elasto.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} elasto.o ../../ch6/poissonfunctions.o

elasto1D: elasto1D.o chkopts
	-${CLINKER} -o elasto1D elasto1D.o ${PETSC_LIB}
	${RM} elasto1D.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ dam elasto elasto1D

