include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

advect: advect.o  chkopts
	-${CLINKER} -o advect advect.o  ${PETSC_LIB}
	${RM} advect.o

both: both.o  chkopts
	-${CLINKER} -o both both.o  ${PETSC_LIB}
	${RM} both.o

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
	-@../testit.sh advect "-da_grid_x 3 -da_grid_y 3 -adv_limiter centered -adv_jac_limiter centered -ts_type cn -ts_adapt_type basic -ts_monitor -ts_dt 0.01 -ts_max_time 0.02 -snes_converged_reason" 1 4


# basic test of diffusion part (NOWIND)
runboth_1:
	-@../testit.sh both "-snes_type ksponly -ksp_converged_reason -bth_problem nowind -bth_eps 1.0 -da_refine 1" 1 1

# GMG+GS test of first-order upwinding for LAYER
runboth_2:
	-@../testit.sh both "-snes_type ksponly -ksp_converged_reason -bth_problem layer -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward -da_refine 2" 1 2

# grid sequence and van Leer limiter for LAYER
runboth_3:
	-@../testit.sh both "-snes_converged_reason -ksp_converged_reason -bth_problem layer -bth_limiter vanleer -snes_grid_sequence 1" 1 3

# GMG with ILU smoothing, van Leer limiter, and -bth_none_on_peclet (thus none always) for GLAZE
runboth_4:
	-@../testit.sh both "-snes_converged_reason -ksp_converged_reason -bth_problem glaze -bth_eps 0.005 -bth_limiter vanleer -bth_none_on_peclet -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -da_refine 1" 1 4

# parallel GMG with ASM+ILU smoothing, centered limiter, -bth_none_on_peclet, and 5x5 coarse grid for LAYER with eps=0.49 (so centered on 9x9, none on 5x5)
runboth_5:
	-@../testit.sh both "-snes_type ksponly -ksp_monitor -bth_problem layer -bth_eps 0.49 -bth_limiter centered -bth_none_on_peclet -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type asm -mg_levels_sub_pc_type ilu -da_refine 2 -pc_mg_levels 2" 2 5

test_advect: runadvect_1 runadvect_2 runadvect_3 runadvect_4

test_both: runboth_1 runboth_2 runboth_3 runboth_4 runboth_5

test: test_advect test_both

# etc

.PHONY: distclean runadvect_1 runadvect_2 runadvect_3 runadvect_4 runboth_1 runboth_2 runboth_3 runboth_4 runboth_5 test_advect test_both test

distclean:
	@rm -f *~ *tmp *.pyc *.dat *.dat.info advect both

