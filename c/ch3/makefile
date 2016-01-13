include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o chkopts
	-${CLINKER} -o poisson poisson.o ${PETSC_KSP_LIB}
	${RM} poisson.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson

