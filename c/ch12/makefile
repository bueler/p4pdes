include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

obstacle: obstacle.o ../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o obstacle obstacle.o ../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} obstacle.o ../ch6/poissonfunctions.o

dnlo: dnlo.o  chkopts
	-${CLINKER} -o dnlo dnlo.o  ${PETSC_LIB}
	${RM} dnlo.o

# testing

runobstacle_1:
	-@../testit.sh obstacle "-da_refine 2 -mat_is_symmetric -snes_monitor_short -ksp_rtol 1.0e-12 -snes_rtol 1.0e-10 -ksp_converged_reason" 1 1

runobstacle_2:
	-@../testit.sh obstacle "-da_refine 3 -snes_monitor_short -pc_type asm -sub_pc_type lu" 4 2

runobstacle_3:
	-@../testit.sh obstacle "-snes_grid_sequence 3 -snes_converged_reason -ksp_type cg -pc_type mg" 1 3

rundnlo_1:
	-@../testit.sh dnlo "-snes_monitor_short -da_refine 1" 1 1

# FIXME  convert these to dnlo tests
runice_1:
	-@../testit.sh ice "-ice_verif -ice_exact_init -da_refine 1 -snes_converged_reason" 1 1

runice_2:
	-@../testit.sh ice "-ice_verif -snes_grid_sequence 1 -snes_converged_reason" 1 2

runice_3:
	-@../testit.sh ice "-ice_verif -da_refine 1 -snes_type vinewtonrsls -snes_converged_reason -ice_check_admissible -ice_exact_init -ice_eps 0.1" 1 3

test_obstacle: runobstacle_1 runobstacle_2 runobstacle_3

test_dnlo: rundnlo_1

test: test_obstacle test_dnlo

# etc

.PHONY: distclean runobstacle_1 runobstacle_2 runobstacle_3 rundnlo_1 test test_obstacle test_dnlo

distclean:
	@rm -f *~ obstacle dnlo *.dat *.dat.info *.pdf *.pyc *tmp
	@rm -f PetscBinaryIO.py petsc_conf.py
	(cd solns; ${MAKE} distclean)

