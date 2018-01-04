include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

minimal: minimal.o ../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o minimal minimal.o ../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} minimal.o ../ch6/poissonfunctions.o

# testing

runminimal_1:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_monitor_short -mse_problem catenoid -mse_catenoid_c 2.0 -da_refine 1" 1 1

runminimal_2:
	-@../testit.sh minimal "-snes_fd_color -mat_is_symmetric 1.0e-7 -mse_q 0.0 -ksp_type cg -ksp_converged_reason -da_refine 2 -mse_problem tent" 1 2

runminimal_3:
	-@../testit.sh minimal "-snes_mf_operator -snes_converged_reason -pc_type mg -da_refine 2 -mse_monitor -mse_quaddegree 2" 2 3

runminimal_4:
	-@../testit.sh minimal "-snes_fd_color -snes_converged_reason -snes_grid_sequence 2 -mse_problem tent" 1 4

test_minimal: runminimal_1 runminimal_2 runminimal_3 runminimal_4

test: test_minimal

# etc

.PHONY: distclean runminimal_1 runminimal_2 runminimal_3 runminimal_4 test test_minimal

distclean:
	@rm -f *~ minimal *tmp

