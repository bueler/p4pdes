include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fish2: fish2.o chkopts
	-${CLINKER} -o fish2 fish2.o ${PETSC_LIB}
	${RM} fish2.o

fish3: fish3.o chkopts
	-${CLINKER} -o fish3 fish3.o ${PETSC_LIB}
	${RM} fish3.o

# testing

runfish2_1:
	-@../testit.sh fish2 "-da_refine 1 -pc_type mg -ksp_converged_reason" 1 1

runfish2_2:
	-@../testit.sh fish2 "-da_refine 4 -pc_type mg -pc_mg_cycle_type w -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -ksp_converged_reason" 2 2

runfish3_1:
	-@../testit.sh fish3 "-ksp_converged_reason -snes_converged_reason -da_refine 1" 1 1

runfish3_2:
	-@../testit.sh fish3 "-snes_fd_color -ksp_rtol 1.0e-12 -pc_type mg -pc_mg_galerkin -ksp_converged_reason -da_refine 2" 2 2

test: runfish2_1 runfish2_2 runfish3_1 runfish3_2

# etc

.PHONY: distclean

distclean:
	@rm -f *~ fish2 fish3

