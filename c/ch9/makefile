include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

obstacle: obstacle.o  chkopts
	-${CLINKER} -o obstacle obstacle.o  ${PETSC_LIB}
	${RM} obstacle.o

# testing

runobstacle_1:
	-@../testit.sh obstacle "-da_refine 3 -snes_monitor_short -ksp_rtol 1.0e-12 -snes_rtol 1.0e-10 -ksp_converged_reason" 1 1

runobstacle_2:
	-@../testit.sh obstacle "-da_refine 4 -snes_monitor_short -pc_type asm -sub_pc_type lu" 4 2

runobstacle_3:
	-@../testit.sh obstacle "-da_refine 3 -mat_is_symmetric" 1 3

test: runobstacle_1 runobstacle_2 runobstacle_3

# etc

.PHONY: distclean

distclean:
	@rm -f *~ obstacle

