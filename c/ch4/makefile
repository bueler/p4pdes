include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

expcircle: expcircle.o chkopts
	-${CLINKER} -o expcircle expcircle.o ${PETSC_LIB}
	${RM} expcircle.o

ecjac: ecjac.o chkopts
	-${CLINKER} -o ecjac ecjac.o ${PETSC_LIB}
	${RM} ecjac.o

reaction: reaction.o chkopts
	-${CLINKER} -o reaction reaction.o ${PETSC_LIB}
	${RM} reaction.o

# testing
runexpcircle_1:
	-@../testit.sh expcircle "-snes_fd" 1 1

runexpcircle_2:
	-@../testit.sh expcircle "-snes_mf" 1 2

runecjac_1:
	-@../testit.sh ecjac "" 1 1

runecjac_2:
	-@../testit.sh ecjac "-snes_mf_operator" 1 2

runreaction_1:
	-@../testit.sh reaction "-snes_converged_reason -da_refine 2" 1 1

runreaction_2:
	-@../testit.sh reaction "-da_grid_x 7" 1 2

runreaction_3:
	-@../testit.sh reaction "-snes_fd_color -da_refine 2" 4 3

runreaction_4:
	-@../testit.sh reaction "-snes_converged_reason -snes_linesearch_type cp" 1 4

test: runexpcircle_1 runexpcircle_2 runecjac_1 runecjac_2 runreaction_1 runreaction_2 runreaction_3 runreaction_4

# etc

.PHONY: distclean runexpcircle_1 runexpcircle_2 runecjac_1 runecjac_2 runreaction_1 runreaction_2 runreaction_3 runreaction_4 test

distclean:
	@rm -f *~ expcircle ecjac reaction

