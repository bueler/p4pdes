include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

well: well.o  chkopts
	-${CLINKER} -o well well.o  ${PETSC_LIB}
	${RM} well.o

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_LIB}
	${RM} stokes.o

# copy of this broken example from petsc commit 55cee6f09eb989b4d30b8ed1c133ceefb891d332
ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_LIB}
	${RM} ex7.o

# testing

runwell_1:
	-@../testit.sh well "-snes_converged_reason -da_refine 1 -well_shorterrors" 1 1

runwell_2:
	-@../testit.sh well "-snes_converged_reason -ksp_converged_reason -ksp_type minres -pc_type none -da_refine 2 -ksp_rtol 1.0e-14 -well_shorterrors" 1 2

runwell_3:
	-@../testit.sh well "-snes_converged_reason -snes_monitor_short -da_refine 1 -well_scheme regular -well_shorterrors" 1 3

test_well: runwell_1 runwell_2 runwell_3

test: test_well


# etc

.PHONY: distclean runwell_1 runwell_2 runwell_3 test_well test

distclean:
	@rm -f *~ *tmp *.dat *.dat.info *.pyc PetscBinaryIO.py petsc_conf.py stokes ex7 well

