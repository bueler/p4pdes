
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

c1matvec: c1matvec.o  chkopts
	-${CLINKER} -o c1matvec c1matvec.o  ${PETSC_KSP_LIB}
	${RM} c1matvec.o

c2convert: c2convert.o readmesh.o  chkopts
	-${CLINKER} -o c2convert c2convert.o readmesh.o  ${PETSC_KSP_LIB}
	${RM} c2convert.o readmesh.o

c2testprealloc: c2testprealloc.o readmesh.o poissontools.o  chkopts
	-${CLINKER} -o c2testprealloc c2testprealloc.o readmesh.o poissontools.o  ${PETSC_KSP_LIB}
	${RM} c2testprealloc.o readmesh.o poissontools.o

c2poisson: c2poisson.o readmesh.o poissontools.o  chkopts
	-${CLINKER} -o c2poisson c2poisson.o readmesh.o poissontools.o  ${PETSC_KSP_LIB}
	${RM} c2poisson.o readmesh.o poissontools.o


.PHONY: distclean

distclean:
	@rm -f *~ ex? ex?? c1poisson
