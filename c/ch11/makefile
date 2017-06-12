include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

obstacle: obstacle.o ../ch6/poissonfunctions.o chkopts
	-${CLINKER} -o obstacle obstacle.o ../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} obstacle.o ../ch6/poissonfunctions.o

ice: ice.o  chkopts
	-${CLINKER} -o ice ice.o  ${PETSC_LIB}
	${RM} ice.o

# testing

runobstacle_1:
	-@../testit.sh obstacle "-da_refine 3 -snes_monitor_short -ksp_rtol 1.0e-12 -snes_rtol 1.0e-10 -ksp_converged_reason" 1 1

runobstacle_2:
	-@../testit.sh obstacle "-da_refine 4 -snes_monitor_short -pc_type asm -sub_pc_type lu" 4 2

runobstacle_3:
	-@../testit.sh obstacle "-da_refine 3 -mat_is_symmetric" 1 3

runice_1:
	-@../testit.sh ice "-da_refine 2 -ice_verif 1 -ice_eps 0.0 -ice_tf 40 -ice_dtinit 10 -ts_type beuler -ts_adapt_type none" 1 1

runice_2:
	-@../testit.sh ice "-da_refine 2 -ice_verif 2 -ice_eps 0.0 -ice_tf 10 -ice_dtinit 3 -snes_fd_color" 2 2

runice_3:
	-@../testit.sh ice "-da_refine 1 -ice_tf 4 -ice_dtinit 1 -ice_maxslide 500 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none" 1 3

runice_4:
	-@../testit.sh ice "-da_refine 2 -ice_tf 20 -ice_dtinit 1 -snes_converged_reason -ts_adapt_clip 0.1,10.0" 1 4

test_obstacle: runobstacle_1 runobstacle_2 runobstacle_3

test_ice: runice_1 runice_2 runice_3 runice_4

test: test_obstacle test_ice

# etc

.PHONY: distclean runobstacle_1 runobstacle_2 runobstacle_3 runice_1 runice_2 runice_3 runice_4 test test_obstacle test_ice

distclean:
	@rm -f *~ obstacle ice *tmp

