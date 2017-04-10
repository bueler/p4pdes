include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

advect: advect.o  chkopts
	-${CLINKER} -o advect advect.o  ${PETSC_LIB}
	${RM} advect.o

ad3: ad3.o  chkopts
	-${CLINKER} -o ad3 ad3.o  ${PETSC_LIB}
	${RM} ad3.o

# testing
# first-order upwinding and nonstandard RK
runadvect_1:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -ts_rk_type 5f -adv_limiter none" 1 1

# vanleer limiter [default] and parallel and beuler and rotation and -snes_fd_color
runadvect_2:
	-@../testit.sh advect "-da_refine 2 -ts_monitor -adv_problem rotation -ts_type beuler -snes_monitor_short -ts_final_time 0.01 -ts_dt 0.01 -snes_rtol 1.0e-4 -snes_fd_color" 2 2

# koren limiter and rightward wind and CN and analytical jacobian
runadvect_3:
	-@../testit.sh advect "-da_refine 1 -ts_monitor -adv_limiter koren -adv_windy 0.0 -ts_final_time 0.02 -ts_dt 0.01 -ts_type cn -snes_converged_reason" 1 3

runad3_1:
	-@../testit.sh ad3 "-da_refine 1 -snes_converged_reason" 1 1

test: runadvect_1 runadvect_2 runadvect_3 runad3_1

# etc

.PHONY: distclean runadvect_1 runadvect_2

distclean:
	@rm -f *~ *tmp advect ad3

