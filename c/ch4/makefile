include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

reaction: reaction.o chkopts
	-${CLINKER} -o reaction reaction.o ${PETSC_SNES_LIB}
	${RM} reaction.o 

# etc

.PHONY: distclean

distclean:
	@rm -f *~ reaction

