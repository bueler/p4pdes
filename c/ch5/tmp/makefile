include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

plapV0: plapV0.o chkopts
	-${CLINKER} -o plapV0 plapV0.o ${PETSC_LIB}
	${RM} plapV0.o

.PHONY: distclean

distclean:
	@rm -f *~ plapV0

