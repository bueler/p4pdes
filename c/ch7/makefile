include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

minimal: minimal.o ../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o minimal minimal.o ../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} minimal.o ../ch6/poissonfunctions.o

biharm: biharm.o ../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o biharm biharm.o ../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} biharm.o ../ch6/poissonfunctions.o

# testing

runminimal_1:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_monitor_short -ms_problem catenoid -ms_catenoid_c 2.0 -da_refine 1" 1 1

runminimal_2:
	-@../testit.sh minimal "-snes_fd_color -mat_is_symmetric 1.0e-7 -ms_q 0.0 -ksp_type cg -ksp_converged_reason -da_refine 2 -ms_problem tent" 1 2

runminimal_3:
	-@../testit.sh minimal "-snes_mf_operator -snes_converged_reason -pc_type mg -snes_grid_sequence 2 -ms_monitor -ms_quaddegree 2" 2 3

runminimal_4:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_grid_sequence 2 -ms_problem tent" 1 4

# monolithic GAMG and FD Jacobian
runbiharm_1:
	-@../testit.sh biharm "-ksp_converged_reason -da_refine 1 -pc_type gamg -snes_fd_color" 1 1

# parallel fieldsplit with defaults (multiplicative fieldsplit, ILU on blocks)
runbiharm_2:
	-@../testit.sh biharm "-ksp_converged_reason -da_refine 1 -pc_type fieldsplit" 2 2

# multiplicative (Gauss-Seidel) fieldsplit with MG preconditioner on blocks
# BUG: needs galerkin; see PETSc issue #307 which reproduces with snes ex19.c
runbiharm_3:
	-@../testit.sh biharm "-ksp_monitor_short -da_refine 2 -pc_type fieldsplit -fieldsplit_v_pc_type mg -fieldsplit_v_pc_mg_galerkin -fieldsplit_v_pc_mg_levels 3 -fieldsplit_v_mg_levels_ksp_type richardson -fieldsplit_u_pc_type mg -fieldsplit_u_pc_mg_galerkin -fieldsplit_u_pc_mg_levels 3 -fieldsplit_u_mg_levels_ksp_type richardson" 1 3

test_minimal: runminimal_1 runminimal_2 runminimal_3 runminimal_4

test_biharm: runbiharm_1 runbiharm_2 runbiharm_3

test: test_minimal test_biharm

# etc

.PHONY: distclean runminimal_1 runminimal_2 runminimal_3 runminimal_4 runbiharm_1 runbiharm_2 runbiharm_3 test test_minimal test_biharm

distclean:
	@rm -f *~ minimal biharm *tmp

