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
	-@../testit.sh advect "-da_refine 1 -ts_monitor -ts_rk_type 5f -adv_limiter none -ts_final_time 1.0" 1 1

# vanleer limiter and parallel and beuler and rotation and -snes_fd_color
runadvect_2:
	-@../testit.sh advect "-da_refine 2 -ts_monitor -adv_limiter vanleer -adv_problem rotation -ts_type beuler -snes_monitor_short -ts_final_time 0.01 -ts_dt 0.01 -snes_rtol 1.0e-4 -snes_fd_color" 2 2

# koren limiter [default] and rightward wind and CN and analytical jacobian
runadvect_3:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -adv_windy 0.0 -ts_final_time 0.02 -ts_dt 0.01 -ts_type cn -snes_converged_reason" 1 3

# eps = 1.0 [default] and centered fluxes and no Jacobian evaluation
runad3_1:
	-@../testit.sh ad3 "-da_refine 1 -snes_converged_reason -ad3_limiter centered -snes_fd_color" 1 1

# eps = 0.01 and upwind and GAMG
runad3_2:
	-@../testit.sh ad3 "-da_refine 1 -ad3_eps 0.01 -ad3_limiter none -pc_type gamg -snes_monitor_short" 1 2

test_advect: runadvect_1 runadvect_2 runadvect_3
test_ad3: runad3_1 runad3_2
test: test_advect test_ad3

# etc

.PHONY: distclean runadvect_1 runadvect_2 runadvect_3 runad3_1

distclean:
	@rm -f *~ *tmp *.pyc advect ad3

