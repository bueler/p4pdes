include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

dam: dam.o ../../ch6/poissonfunctions.o
	-${CLINKER} -o dam dam.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} dam.o ../../ch6/poissonfunctions.o

elasto: elasto.o ../../ch6/poissonfunctions.o
	-${CLINKER} -o elasto elasto.o ../../ch6/poissonfunctions.o ${PETSC_LIB}
	${RM} elasto.o ../../ch6/poissonfunctions.o

# testing

rundam_1:
	-@../../testit.sh dam "-snes_grid_sequence 2 -snes_converged_reason -pc_type mg -mg_levels_ksp_type richardson" 1 1

runelasto_1:
	-@../../testit.sh elasto "-snes_grid_sequence 2 -snes_converged_reason -pc_type mg -mg_levels_ksp_type richardson" 1 1

test_dam: rundam_1

test_elasto: runelasto_1

test: test_dam test_elasto

.PHONY: clean distclean rundam_1 runelasto_1 test test_dam test_elasto

distclean: clean

clean::
	@rm -f *~ *tmp dam elasto
