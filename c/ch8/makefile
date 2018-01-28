include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

phelm: phelm.o chkopts
	-${CLINKER} -o phelm phelm.o ${PETSC_LIB}
	${RM} phelm.o

# testing
runphelm_1:
	-@../testit.sh phelm "-ph_no_gradient -snes_fd_function -snes_fd -snes_converged_reason" 1 1

runphelm_2:
	-@../testit.sh phelm "-ph_problem constant -snes_type ksponly -ksp_converged_reason -ksp_type cg -pc_type mg -da_refine 3" 1 2

runphelm_3:
	-@../testit.sh phelm "-ph_p 1.5 -ph_eps 1.0e-3 -ph_no_objective -ph_quadpts 1 -ph_exact_init -da_refine 1 -snes_converged_reason" 1 3

runphelm_4:
	-@../testit.sh phelm "-ph_no_objective -snes_fd_color -snes_converged_reason -da_refine 1" 2 4

runphelm_5:
	-@../testit.sh phelm "-ph_view_f -ph_p 1.5" 1 5  # generates nan

# FIXME need -snes_grid_sequence -pc_type gamg test (?)

test_phelm: runphelm_1 runphelm_2 runphelm_3 runphelm_4 runphelm_5

test: test_phelm

# etc

.PHONY: distclean runphelm_1 runphelm_2 runphelm_3 runphelm_4 runphelm_5 test test_phelm

distclean:
	@rm -f *~ phelm
	(cd solns/; ${MAKE} distclean);

