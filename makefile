
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

c1poisson: c1poisson.o  chkopts
	-${CLINKER} -o c1poisson c1poisson.o  ${PETSC_KSP_LIB}
	${RM} c1poisson.o


.PHONY: distclean

distclean:
	@rm -f *~ ex? ex?? c1poisson
