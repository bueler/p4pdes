include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fish1: fish1.o chkopts
	-${CLINKER} -o fish1 fish1.o ${PETSC_LIB}
	${RM} fish1.o

fish2: fish2.o chkopts
	-${CLINKER} -o fish2 fish2.o ${PETSC_LIB}
	${RM} fish2.o

fish3: fish3.o chkopts
	-${CLINKER} -o fish3 fish3.o ${PETSC_LIB}
	${RM} fish3.o

minimal: minimal.o chkopts
	-${CLINKER} -o minimal minimal.o ${PETSC_LIB}
	${RM} minimal.o

# testing

runfish1_1:
	-@../testit.sh fish1 "-da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor_short -ksp_converged_reason" 1 1

runfish1_2:
	-@../testit.sh fish1 "-da_refine 1 -mat_is_symmetric 1.0e-5 -snes_fd_color" 1 2

runfish2_1:
	-@../testit.sh fish2 "-da_refine 1 -pc_type mg -ksp_converged_reason" 1 1

runfish2_2:
	-@../testit.sh fish2 "-da_refine 4 -pc_type mg -pc_mg_cycle_type w -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -ksp_converged_reason" 2 2

runfish2_3:
	-@../testit.sh fish2 "-da_refine 3 -mat_is_symmetric" 1 3

runfish2_4:
	-@../testit.sh fish2 "-da_refine 2 -mat_is_symmetric 1.0e-5 -snes_fd_color" 1 4

runfish3_1:
	-@../testit.sh fish3 "-ksp_converged_reason -snes_converged_reason -da_refine 1" 1 1

runfish3_2:
	-@../testit.sh fish3 "-snes_fd_color -ksp_rtol 1.0e-12 -pc_type mg -pc_mg_galerkin -ksp_converged_reason -da_refine 2" 2 2

runfish3_3:
	-@../testit.sh fish3 "-da_refine 2 -mat_is_symmetric" 1 3

runfish3_4:
	-@../testit.sh fish3 "-da_refine 3 -mat_is_symmetric 1.0e-5 -snes_fd_color" 1 4

runminimal_1:
	-@../testit.sh minimal "-snes_fd_color -da_refine 1 -snes_converged_reason" 1 1

runminimal_2:
	-@../testit.sh minimal "-snes_fd_color -da_refine 2 -mat_is_symmetric 1.0e-7 -mse_laplace -ksp_type cg -ksp_converged_reason" 1 2

runminimal_3:
	-@../testit.sh minimal "-snes_mf_operator -da_refine 2 -snes_monitor_short" 2 3

runminimal_4:
	-@../testit.sh minimal "-snes_fd_color -da_refine 2 -snes_monitor_short -pc_type mg" 2 4

test_fish1: runfish1_1 runfish1_2

test_fish2: runfish2_1 runfish2_2 runfish2_3 runfish2_4

test_fish3: runfish3_1 runfish3_2 runfish3_3 runfish3_4

test_minimal: runminimal_1 runminimal_2 runminimal_3 runminimal_4

test: test_fish1 test_fish2 test_fish3 test_minimal

# etc

.PHONY: distclean

distclean:
	@rm -f *~ fish1 fish2 fish3 minimal

