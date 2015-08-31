include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o structuredpoisson.o chkopts
	-${CLINKER} -o poisson poisson.o structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} poisson.o structuredpoisson.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson

