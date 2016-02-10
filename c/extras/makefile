include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

riemann: riemann.o  chkopts
	-${CLINKER} -o riemann riemann.o  ${PETSC_KSP_LIB}
	${RM} riemann.o

testprealloc: testprealloc.o ch10/readmesh.o chkopts
	-${CLINKER} -o testprealloc testprealloc.o ch10/readmesh.o ${PETSC_KSP_LIB}
	${RM} testprealloc.o ch10/readmesh.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ riemann testprealloc

