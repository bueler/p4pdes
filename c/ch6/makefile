include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fish: fish.o poissonfunctions.o chkopts
	-${CLINKER} -o fish fish.o poissonfunctions.o ${PETSC_LIB}
	${RM} fish.o poissonfunctions.o

minimal: minimal.o poissonfunctions.o chkopts
	-${CLINKER} -o minimal minimal.o poissonfunctions.o ${PETSC_LIB}
	${RM} minimal.o poissonfunctions.o

# testing

runfish_1:
	-@../testit.sh fish "-fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor_short -ksp_converged_reason" 1 1

runfish_2:
	-@../testit.sh fish "-fsh_dim 1 -fsh_problem manupoly -da_refine 1 -mat_is_symmetric 1.0e-5 -snes_fd_color" 1 2

runfish_3:
	-@../testit.sh fish "-fsh_dim 2 -fsh_initial_gonboundary false -da_refine 1 -pc_type mg -ksp_converged_reason" 1 3

runfish_4:
	-@../testit.sh fish "-fsh_dim 2 -da_refine 3 -pc_type mg -pc_mg_cycle_type w -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -ksp_converged_reason" 2 4

runfish_5:
	-@../testit.sh fish "-fsh_dim 2 -da_refine 3 -mat_is_symmetric 1.0e-7 -snes_fd_color" 1 5

runfish_6:
	-@../testit.sh fish "-fsh_dim 3 -da_refine 2 -fsh_problem manupoly -ksp_converged_reason -fsh_cx 0.01 -fsh_cy 2 -fsh_cz 100" 1 6

runfish_7:
	-@../testit.sh fish "-fsh_dim 3 -fsh_problem manupoly -snes_fd_color -ksp_rtol 1.0e-12 -pc_type mg -pc_mg_galerkin -ksp_converged_reason -da_refine 2" 2 7

runfish_8:
	-@../testit.sh fish "-fsh_dim 3 -da_refine 2 -mat_is_symmetric 1.0e-7 -snes_fd_color" 1 8

runminimal_1:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_monitor_short -mse_problem catenoid -mse_catenoid_c 2.0 -da_refine 1" 1 1

runminimal_2:
	-@../testit.sh minimal "-snes_fd_color -mat_is_symmetric 1.0e-7 -mse_q 0.0 -ksp_type cg -ksp_converged_reason -da_refine 2 -mse_problem tent" 1 2

runminimal_3:
	-@../testit.sh minimal "-snes_mf_operator -snes_converged_reason -pc_type mg -da_refine 2 -mse_monitor" 2 3

runminimal_4:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_grid_sequence 2 -mse_problem tent" 1 4

test_fish: runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6 runfish_7 runfish_8

test_minimal: runminimal_1 runminimal_2 runminimal_3 runminimal_4

test: test_fish test_minimal

# etc

.PHONY: distclean runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6 runfish_7 runfish_8 runminimal_1 runminimal_2 runminimal_3 test test_fish test_minimal

distclean:
	@rm -f *~ fish minimal *tmp

