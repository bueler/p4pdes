include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson1D: poisson1D.o chkopts
	-${CLINKER} -o poisson1D poisson1D.o ${PETSC_KSP_LIB}
	${RM} poisson1D.o

poisson3D: poisson3D.o chkopts
	-${CLINKER} -o poisson3D poisson3D.o ${PETSC_KSP_LIB}
	${RM} poisson3D.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson1D poisson3D

