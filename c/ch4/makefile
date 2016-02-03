include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

expcircle: expcircle.o chkopts
	-${CLINKER} -o expcircle expcircle.o ${PETSC_LIB}
	${RM} expcircle.o

ecjacobian: ecjacobian.o chkopts
	-${CLINKER} -o ecjacobian ecjacobian.o ${PETSC_LIB}
	${RM} ecjacobian.o

reaction: reaction.o chkopts
	-${CLINKER} -o reaction reaction.o ${PETSC_LIB}
	${RM} reaction.o

# testing
runexpcircle_1:
	-@../testit.sh expcircle "-snes_fd" 1 1

runexpcircle_2:
	-@../testit.sh expcircle "-snes_mf" 1 2

runecjacobian_1:
	-@../testit.sh ecjacobian "" 1 1

runecjacobian_2:
	-@../testit.sh ecjacobian "-snes_mf_operator" 1 2

runreaction_1:
	-@../testit.sh reaction "-snes_converged_reason -da_refine 2" 1 1

runreaction_2:
	-@../testit.sh reaction "-da_grid_x 7" 1 2

runreaction_3:
	-@../testit.sh reaction "-snes_fd_color -da_refine 2" 4 3

runreaction_4:
	-@../testit.sh reaction "-snes_converged_reason -snes_linesearch_type cp" 1 4

# etc

.PHONY: distclean

distclean:
	@rm -f *~ expcircle ecjacobian reaction

