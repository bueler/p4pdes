include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

phelm: phelm.o chkopts
	-${CLINKER} -o phelm phelm.o ${PETSC_LIB}
	${RM} phelm.o

plap: plap.o chkopts
	-${CLINKER} -o plap plap.o ${PETSC_LIB}
	${RM} plap.o

# testing
runphelm_1:
	-@../testit.sh phelm "-ph_p 2.0 -ph_no_residual -snes_fd_function -snes_fd -snes_converged_reason" 1 1

runphelm_2:
	-@../testit.sh phelm "-ph_p 2.0 -snes_converged_reason -ksp_type cg -pc_type mg -da_refine 3" 1 2

runphelm_3:
	-@../testit.sh phelm "-ph_no_objective -ph_quaddegree 1 -ph_exact_init -da_refine 1 -snes_converged_reason" 1 3

# FIXME need -snes_grid_sequence -pc_type gamg test (?)
# FIXME need parallel test of phelm

runplap_1:
	-@../testit.sh plap "-plap_no_residual -snes_fd_function -snes_fd_color -snes_converged_reason" 1 1

runplap_2:
	-@../testit.sh plap "-snes_converged_reason -ksp_type cg -pc_type icc -da_refine 3" 1 2

runplap_3:
	-@../testit.sh plap "-snes_fd_color -snes_converged_reason -da_refine 2 -snes_linesearch_type cp" 4 3

runplap_4:
	-@../testit.sh plap "-plap_quaddegree 1 -snes_converged_reason" 1 4

#runplap_5:  (deactivated because it is slow)
#	-@../testit.sh plap "-plap_no_residual -snes_fd_function -snes_type ncg -snes_converged_reason" 1 5

test_phelm: runphelm_1 runphelm_2 runphelm_3

test_plap: runplap_1 runplap_2 runplap_3 runplap_4

test: test_phelm test_plap

# etc

.PHONY: distclean runphelm_1 runphelm_2 runphelm_3 runplap_1 runplap_2 runplap_3 runplap_4 test test_phelm test_plap

distclean:
	@rm -f *~ phelm plap *tmp

