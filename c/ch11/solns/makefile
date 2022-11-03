include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

both1d: both1d.o
	-${CLINKER} -o both1d both1d.o  ${PETSC_LIB}
	${RM} both1d.o

# testing

runboth1d_1:
	-@../../testit.sh both1d "-b1_eps 0.1 -da_refine 1 -ksp_view_mat ::ascii_dense -snes_monitor_solution -snes_fd_color" 1 1

runboth1d_2:
	-@../../testit.sh both1d "-da_refine 1 -snes_test_jacobian" 1 2

runboth1d_3:
	-@../../testit.sh both1d "-da_refine 2 -snes_type ksponly -ksp_monitor_short -pc_type asm -sub_pc_type sor -sub_pc_sor_forward" 2 3

runboth1d_4:
	-@../../testit.sh both1d "-da_refine 1 -snes_monitor_short -b1_limiter vanleer -b1_jac_limiter none -snes_mf_operator" 1 4

runboth1d_5:
	-@../../testit.sh both1d "-da_refine 1 -snes_type ksponly -b1_limiter centered -b1_jac_limiter centered -snes_test_jacobian" 1 5

test_both1d: runboth1d_1 runboth1d_2 runboth1d_3 runboth1d_4 runboth1d_5

test: test_both1d

.PHONY: clean distclean runboth1d_1 runboth1d_2 runboth1d_3 runboth1d_4 runboth1d_5 test_both1d test

distclean: clean

clean::
	@rm -f *~ *tmp both1d
