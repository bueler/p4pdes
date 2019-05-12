include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

advect: advect.o  chkopts
	-${CLINKER} -o advect advect.o  ${PETSC_LIB}
	${RM} advect.o

both1: both1.o  chkopts
	-${CLINKER} -o both1 both1.o  ${PETSC_LIB}
	${RM} both1.o

both2: both2.o  chkopts
	-${CLINKER} -o both2 both2.o  ${PETSC_LIB}
	${RM} both2.o

# testing

# first-order upwinding and nonstandard RK and evaluation of error
runadvect_1:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -ts_rk_type 5f -adv_limiter none -ts_max_time 0.1" 1 1

# vanleer limiter and parallel and beuler and rotation and -snes_fd_color
runadvect_2:
	-@../testit.sh advect "-da_refine 2 -ts_monitor -adv_limiter vanleer -adv_problem rotation -ts_type beuler -snes_monitor_short -ts_max_time 0.02 -ts_dt 0.02 -snes_rtol 1.0e-4 -snes_fd_color" 2 2

# koren limiter [default] and rightward wind and CN and analytical (upwind) jacobian
runadvect_3:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -adv_windy 0.0 -ts_max_time 0.02 -ts_dt 0.01 -ts_type cn -snes_converged_reason" 1 3

# centered limiter and analytical (centered) jacobian; compare with -snes_fd
runadvect_4:
	-@../testit.sh advect "-da_grid_x 3 -da_grid_y 3 -adv_limiter centered -adv_jacobian centered -ts_type cn -ts_adapt_type basic -ts_monitor -ts_dt 0.01 -ts_max_time 0.02 -snes_converged_reason" 1 4

runboth1_1:
	-@../testit.sh both1 "-b1_eps 0.1 -da_refine 1 -ksp_view_mat ::ascii_dense -snes_monitor_solution" 1 1


test_advect: runadvect_1 runadvect_2 runadvect_3 runadvect_4

test_both1: runboth1_1

test: test_advect test_both1

# etc

.PHONY: distclean runadvect_1 runadvect_2 runadvect_3 runadvect_4 runboth1_1 test_advect test_both1 test

distclean:
	@rm -f *~ *tmp *.pyc advect both1 both2

