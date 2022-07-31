include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratu2D: bratu2D.o ../../ch6/poissonfunctions.o
	-${CLINKER} -o bratu2D bratu2D.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} bratu2D.o ../../ch6/poissonfunctions.o

# testing

runbratu2D_1:
	-@../../testit.sh bratu2D "-lb_exact -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 2 -snes_monitor_short" 1 1

test_bratu2D: runbratu2D_1

test: test_bratu2D

# etc

.PHONY: distclean test test_bratu2D runbratu2D_1

distclean:
	@rm -f bratu2D *tmp
