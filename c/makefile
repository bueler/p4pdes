
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

c1e: c1e.o  chkopts
	-${CLINKER} -o c1e c1e.o  ${PETSC_LIB}
	${RM} c1e.o

c1matvec: c1matvec.o  chkopts
	-${CLINKER} -o c1matvec c1matvec.o  ${PETSC_KSP_LIB}
	${RM} c1matvec.o

testmatcreate: testmatcreate.o  chkopts
	-${CLINKER} -o testmatcreate testmatcreate.o  ${PETSC_KSP_LIB}
	${RM} testmatcreate.o

testts: testts.o  chkopts
	-${CLINKER} -o testts testts.o  ${PETSC_KSP_LIB}
	${RM} testts.o

c2poisson: c2poisson.o structuredpoisson.o chkopts
	-${CLINKER} -o c2poisson c2poisson.o structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} c2poisson.o structuredpoisson.o

c2heat: c2heat.o structuredpoisson.o chkopts
	-${CLINKER} -o c2heat c2heat.o structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} c2heat.o structuredpoisson.o

c3convert: c3convert.o readmesh.o  chkopts
	-${CLINKER} -o c3convert c3convert.o readmesh.o  ${PETSC_KSP_LIB}
	${RM} c3convert.o readmesh.o

c3poisson: c3poisson.o readmesh.o poissontools.o chkopts
	-${CLINKER} -o c3poisson c3poisson.o readmesh.o poissontools.o ${PETSC_KSP_LIB}
	${RM} c3poisson.o readmesh.o poissontools.o

c4poisson: c4poisson.o structuredpoisson.o  chkopts
	-${CLINKER} -o c4poisson c4poisson.o structuredpoisson.o  ${PETSC_KSP_LIB}
	${RM} c4poisson.o structuredpoisson.o

c7fosls: c7fosls.o  chkopts
	-${CLINKER} -o c7fosls c7fosls.o  ${PETSC_KSP_LIB}
	${RM} c7fosls.o

c7stokes: c7stokes.o  chkopts
	-${CLINKER} -o c7stokes c7stokes.o  ${PETSC_SNES_LIB}
	${RM} c7stokes.o

c9obstacle: c9obstacle.o  chkopts
	-${CLINKER} -o c9obstacle c9obstacle.o  ${PETSC_SNES_LIB}
	${RM} c9obstacle.o


.PHONY: distclean

distclean:
	@rm -f *~ c1e c1matvec c2poisson c2heat c3convert c3poisson c4poisson c7fosls c7stokes c9obstacle
	@rm -f mesh.?.* bump.?.* square.?.*
