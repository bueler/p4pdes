include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_LIB}
	${RM} stokes.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ stokes

