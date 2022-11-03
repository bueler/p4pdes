include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

expcircle: expcircle.o
	-${CLINKER} -o expcircle expcircle.o ${PETSC_LIB}
	${RM} expcircle.o

ecjac: ecjac.o
	-${CLINKER} -o ecjac ecjac.o ${PETSC_LIB}
	${RM} ecjac.o

reaction: reaction.o
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
	-@../testit.sh reaction "-snes_converged_reason -da_refine 1" 1 1

runreaction_2:
	-@../testit.sh reaction "-da_grid_x 7 -mat_is_symmetric" 1 2

runreaction_3:
	-@../testit.sh reaction "-snes_fd_color -da_refine 1 -snes_view" 4 3

runreaction_4:
	-@../testit.sh reaction "-snes_converged_reason -snes_linesearch_type cp" 1 4

runreaction_5:
	-@../testit.sh reaction "-snes_test_jacobian" 1 5

runreaction_6:
	-@../testit.sh reaction "-snes_converged_reason -rct_noRinJ" 1 6

test_expcircle: runexpcircle_1 runexpcircle_2

test_ecjac: runecjac_1 runecjac_2

test_reaction: runreaction_1 runreaction_2 runreaction_3 runreaction_4 runreaction_5 runreaction_6

test: test_expcircle test_ecjac test_reaction

.PHONY: clean distclean runexpcircle_1 runexpcircle_2 runecjac_1 runecjac_2 runreaction_1 runreaction_2 runreaction_3 runreaction_4 runreaction_5 runreaction_6 test test_expcircle test_ecjac test_reaction

distclean: clean

clean::
	@rm -f *~ expcircle ecjac reaction *tmp

