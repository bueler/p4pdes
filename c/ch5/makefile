include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


poisson: poisson.o ../ch3/structuredpoisson.o  chkopts
	-${CLINKER} -o poisson poisson.o ../ch3/structuredpoisson.o  ${PETSC_KSP_LIB}
	${RM} poisson.o ../ch3/structuredpoisson.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson

