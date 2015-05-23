include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# all codes in text "PETSc for PDEs":

c1e: c1e.o  chkopts
	-${CLINKER} -o c1e c1e.o  ${PETSC_LIB}
	${RM} c1e.o

c1vecmatksp: c1vecmatksp.o  chkopts
	-${CLINKER} -o c1vecmatksp c1vecmatksp.o  ${PETSC_KSP_LIB}
	${RM} c1vecmatksp.o

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

c7riemann: c7riemann.o  chkopts
	-${CLINKER} -o c7riemann c7riemann.o  ${PETSC_KSP_LIB}
	${RM} c7riemann.o

c7fosls: c7fosls.o  chkopts
	-${CLINKER} -o c7fosls c7fosls.o  ${PETSC_KSP_LIB}
	${RM} c7fosls.o

c7stokes: c7stokes.o  chkopts
	-${CLINKER} -o c7stokes c7stokes.o  ${PETSC_SNES_LIB}
	${RM} c7stokes.o

c9obstacle: c9obstacle.o  chkopts
	-${CLINKER} -o c9obstacle c9obstacle.o  ${PETSC_SNES_LIB}
	${RM} c9obstacle.o


# test codes

testmatcreate: testmatcreate.o  chkopts
	-${CLINKER} -o testmatcreate testmatcreate.o  ${PETSC_KSP_LIB}
	${RM} testmatcreate.o

testts: testts.o  chkopts
	-${CLINKER} -o testts testts.o  ${PETSC_KSP_LIB}
	${RM} testts.o


# exercise solutions

e1balanced: e1balanced.o  chkopts
	-${CLINKER} -o e1balanced e1balanced.o  ${PETSC_LIB}
	${RM} e1balanced.o

e1matvecn: e1matvecn.o  chkopts
	-${CLINKER} -o e1matvecn e1matvecn.o  ${PETSC_KSP_LIB}
	${RM} e1matvecn.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ c1e c1vecmatksp c1tri c2poisson c2heat c3convert c3poisson c4poisson c7fosls c7stokes c9obstacle
	@rm -f mesh.?.* bump.?.* square.?.*
