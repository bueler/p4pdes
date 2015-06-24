include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

expcircle: expcircle.o chkopts
	-${CLINKER} -o expcircle expcircle.o ${PETSC_SNES_LIB}
	${RM} expcircle.o

expcircleJAC: expcircleJAC.o chkopts
	-${CLINKER} -o expcircleJAC expcircleJAC.o ${PETSC_SNES_LIB}
	${RM} expcircleJAC.o

reaction: reaction.o chkopts
	-${CLINKER} -o reaction reaction.o ${PETSC_SNES_LIB}
	${RM} reaction.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ expcircle expcircleJAC reaction

