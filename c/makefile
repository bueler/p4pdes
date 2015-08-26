include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# all codes in text "PETSc for PDEs":

heat: heat.o ch3/structuredpoisson.o chkopts
	-${CLINKER} -o heat heat.o ch3/structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} heat.o ch3/structuredpoisson.o

riemann: riemann.o  chkopts
	-${CLINKER} -o riemann riemann.o  ${PETSC_KSP_LIB}
	${RM} riemann.o

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_SNES_LIB}
	${RM} stokes.o

# test codes

testprealloc: testprealloc.o ch4/readmesh.o chkopts
	-${CLINKER} -o testprealloc testprealloc.o ch4/readmesh.o ${PETSC_KSP_LIB}
	${RM} testprealloc.o ch4/readmesh.o

testts: testts.o  chkopts
	-${CLINKER} -o testts testts.o  ${PETSC_KSP_LIB}
	${RM} testts.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ heat stokes riemann testprealloc testts

