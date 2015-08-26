include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

obstacle: obstacle.o  chkopts
	-${CLINKER} -o obstacle obstacle.o  ${PETSC_SNES_LIB}
	${RM} obstacle.o

# test codes

# etc

.PHONY: distclean

distclean:
	@rm -f *~ obstacle

