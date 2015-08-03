include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

expcircle: expcircle.o chkopts
	-${CLINKER} -o expcircle expcircle.o ${PETSC_SNES_LIB}
	${RM} expcircle.o

ecjacobian: ecjacobian.o chkopts
	-${CLINKER} -o ecjacobian ecjacobian.o ${PETSC_SNES_LIB}
	${RM} ecjacobian.o

reaction: reaction.o chkopts
	-${CLINKER} -o reaction reaction.o ${PETSC_SNES_LIB}
	${RM} reaction.o

# solutions to exercises

e3atan: e3atan.o chkopts
	-${CLINKER} -o e3atan e3atan.o ${PETSC_SNES_LIB}
	${RM} e3atan.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ expcircle ecjacobian reaction e3atan

