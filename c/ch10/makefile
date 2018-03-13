include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

advect: advect.o  chkopts
	-${CLINKER} -o advect advect.o  ${PETSC_LIB}
	${RM} advect.o

ad3: ad3.o  chkopts
	-${CLINKER} -o ad3 ad3.o  ${PETSC_LIB}
	${RM} ad3.o

# testing

# first-order upwinding and nonstandard RK and evaluation of error
runadvect_1:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -ts_rk_type 5f -adv_limiter none -ts_final_time 0.1" 1 1

# vanleer limiter and parallel and beuler and rotation and -snes_fd_color
runadvect_2:
	-@../testit.sh advect "-da_refine 2 -ts_monitor -adv_limiter vanleer -adv_problem rotation -ts_type beuler -snes_monitor_short -ts_final_time 0.02 -ts_dt 0.02 -snes_rtol 1.0e-4 -snes_fd_color" 2 2

# koren limiter [default] and rightward wind and CN and analytical (upwind) jacobian
runadvect_3:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -adv_windy 0.0 -ts_final_time 0.02 -ts_dt 0.01 -ts_type cn -snes_converged_reason" 1 3

# centered limiter and analytical (centered) jacobian; compare with -snes_fd
runadvect_4:
	-@../testit.sh advect "-da_grid_x 3 -da_grid_y 3 -adv_limiter centered -adv_jacobian centered -ts_type cn -ts_adapt_type basic -ts_monitor -ts_dt 0.01 -ts_final_time 0.02 -snes_converged_reason" 1 4


# defaults: eps = 1.0, centered fluxes, -snes_fd_color
runad3_1:
	-@../testit.sh ad3 "-snes_converged_reason" 1 1

# eps = 0.01 and upwind and GAMG
runad3_2:
	-@../testit.sh ad3 "-ad3_eps 0.01 -ad3_limiter none -pc_type gamg -snes_converged_reason" 1 2


test_advect: runadvect_1 runadvect_2 runadvect_3 runadvect_4

test_ad3: runad3_1 runad3_2

test: test_advect test_ad3

# etc

.PHONY: distclean runadvect_1 runadvect_2 runadvect_3 runadvect_4 runad3_1 runad3_2 test_advect test_ad3 test

distclean:
	@rm -f *~ *tmp *.pyc advect ad3

