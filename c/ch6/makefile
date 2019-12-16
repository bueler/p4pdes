include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

fish: fish.o poissonfunctions.o
	-${CLINKER} -o fish fish.o poissonfunctions.o ${PETSC_LIB}
	${RM} fish.o poissonfunctions.o

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

test_fish: runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6 runfish_7 runfish_8

test: test_fish

# etc

.PHONY: distclean runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6 runfish_7 runfish_8 test test_fish

distclean:
	@rm -f *~ fish *tmp

