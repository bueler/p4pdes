include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_LIB}
	${RM} stokes.o

pattern: pattern.o  chkopts
	-${CLINKER} -o pattern pattern.o  ${PETSC_LIB}
	${RM} pattern.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ pattern stokes

