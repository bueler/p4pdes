include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_LIB}
	${RM} stokes.o

patterns: patterns.o  chkopts
	-${CLINKER} -o patterns patterns.o  ${PETSC_LIB}
	${RM} patterns.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ patterns stokes

