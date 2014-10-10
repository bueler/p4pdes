
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

c1matvec: c1matvec.o  chkopts
	-${CLINKER} -o c1matvec c1matvec.o  ${PETSC_KSP_LIB}
	${RM} c1matvec.o

c2convert: c2convert.o readmesh.o  chkopts
	-${CLINKER} -o c2convert c2convert.o readmesh.o  ${PETSC_KSP_LIB}
	${RM} c2convert.o readmesh.o

c2poisson: c2poisson.o readmesh.o poissontools.o chkopts
	-${CLINKER} -o c2poisson c2poisson.o readmesh.o poissontools.o ${PETSC_KSP_LIB}
	${RM} c2poisson.o readmesh.o poissontools.o


.PHONY: distclean

distclean:
	@rm -f *~ c1matvec c2convert c2poisson
	@rm -f mesh.?.* bump.?.* square.?.*
