include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o structuredpoisson.o chkopts
	-${CLINKER} -o poisson poisson.o structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} poisson.o structuredpoisson.o


e1poisson1D: e1poisson1D.o chkopts
	-${CLINKER} -o e1poisson1D e1poisson1D.o ${PETSC_KSP_LIB}
	${RM} e1poisson1D.o 

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson e1poisson1D

