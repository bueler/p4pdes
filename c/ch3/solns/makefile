include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson1D: poisson1D.o
	-${CLINKER} -o poisson1D poisson1D.o ${PETSC_KSP_LIB}
	${RM} poisson1D.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson1D

