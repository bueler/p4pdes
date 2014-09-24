
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

c1matvec: c1matvec.o  chkopts
	-${CLINKER} -o c1matvec c1matvec.o  ${PETSC_KSP_LIB}
	${RM} c1matvec.o

c2triangle: c2triangle.o  chkopts
	-${CLINKER} -o c2triangle c2triangle.o  ${PETSC_KSP_LIB}
	${RM} c2triangle.o

c2prealloc: c2prealloc.o  chkopts
	-${CLINKER} -o c2prealloc c2prealloc.o  ${PETSC_KSP_LIB}
	${RM} c2prealloc.o

c2poisson: c2poisson.o  chkopts
	-${CLINKER} -o c2poisson c2poisson.o  ${PETSC_KSP_LIB}
	${RM} c2poisson.o


.PHONY: distclean

distclean:
	@rm -f *~ ex? ex?? c1poisson
