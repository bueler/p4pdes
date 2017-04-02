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
	-@../testit.sh advect "-da_refine 1 -ts_monitor -ts_rk_type 5f -adv_firstorder" 1 1

# parallel and implicit and circlewind
runadvect_2:
	-@../testit.sh advect "-da_refine 2 -ts_monitor -adv_circlewind -adv_conex 0.3 -adv_coney 0.3 -ts_type beuler -snes_monitor_short -ts_final_time 0.01 -ts_dt 0.01 -snes_rtol 1.0e-4" 2 2

test: runadvect_1 runadvect_2

# etc

.PHONY: distclean runadvect_1 runadvect_2

distclean:
	@rm -f *~ *tmp advect ad3

