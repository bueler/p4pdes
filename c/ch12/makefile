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
	-@../testit.sh obstacle "-da_refine 2 -mat_is_symmetric -snes_monitor_short -ksp_rtol 1.0e-12 -snes_rtol 1.0e-10 -ksp_converged_reason" 1 1

runobstacle_2:
	-@../testit.sh obstacle "-da_refine 3 -snes_monitor_short -pc_type asm -sub_pc_type lu" 4 2

runobstacle_3:
	-@../testit.sh obstacle "-snes_grid_sequence 3 -snes_converged_reason -ksp_type cg -pc_type mg" 1 3

test_obstacle: runobstacle_1 runobstacle_2 runobstacle_3

test: test_obstacle

# etc

.PHONY: distclean runobstacle_1 runobstacle_2 runobstacle_3 test test_obstacle

distclean:
	@rm -f *~ obstacle ice *.pyc *tmp
	@rm -f PetscBinaryIO.py petsc_conf.py
	(cd solns; ${MAKE} distclean)

