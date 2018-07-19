include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

riemann: riemann.o  chkopts
	-${CLINKER} -o riemann riemann.o  ${PETSC_KSP_LIB}
	${RM} riemann.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ riemann

