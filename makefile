
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

c1matvec: c1matvec.o  chkopts
	-${CLINKER} -o c1matvec c1matvec.o  ${PETSC_KSP_LIB}
	${RM} c1matvec.o

c1prealloc: c1prealloc.o  chkopts
	-${CLINKER} -o c1prealloc c1prealloc.o  ${PETSC_KSP_LIB}
	${RM} c1prealloc.o

c1poisson: c1poisson.o  chkopts
	-${CLINKER} -o c1poisson c1poisson.o  ${PETSC_KSP_LIB}
	${RM} c1poisson.o


.PHONY: distclean

distclean:
	@rm -f *~ ex? ex?? c1poisson
