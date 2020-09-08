include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

ode: ode.o
	-${CLINKER} -o ode ode.o  ${PETSC_LIB}
	${RM} ode.o

odejac: odejac.o
	-${CLINKER} -o odejac odejac.o  ${PETSC_LIB}
	${RM} odejac.o

heat: heat.o
	-${CLINKER} -o heat heat.o  ${PETSC_LIB}
	${RM} heat.o

pattern: pattern.o
	-${CLINKER} -o pattern pattern.o  ${PETSC_LIB}
	${RM} pattern.o

# use this target to create symbolic links to PETSc binary files scripts
petscPyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py


# testing
runode_1:
	-@../testit.sh ode "-ts_init_time 19.0 -ts_dt 0.01 -ts_max_time 20.0" 1 1

runode_2:
	-@../testit.sh ode "-ts_type cn -snes_fd -ts_max_time 1.0" 1 2

runode_3:
	-@../testit.sh ode "-ts_monitor -ts_max_time 1.0" 1 3

runodejac_1:
	-@../testit.sh odejac "-ts_max_time 1.0" 1 1

runodejac_2:
	-@../testit.sh odejac "-ts_monitor -ts_max_time 1.0 -ts_type rk" 1 2

runheat_1:
	-@../testit.sh heat "-da_refine 1 -ts_monitor -ts_type beuler" 1 1

runheat_2:
	-@../testit.sh heat "-da_refine 1 -ts_monitor -ts_type rk -ts_max_time 0.01" 2 2

runpattern_1:
	-@../testit.sh pattern "-da_grid_x 4 -da_grid_y 4 -da_refine 2 -ts_monitor" 1 1   # refinement of 1 misses initial condition

runpattern_2:
	-@../testit.sh pattern "-da_refine 2 -ts_monitor -ts_dt 1 -ts_max_time 1 -ts_type beuler -pc_type mg -snes_converged_reason -ksp_converged_reason -snes_rtol 1.0e-1 -ptn_no_rhsjacobian" 1 2

runpattern_3:
	-@../testit.sh pattern "-da_refine 2 -ts_monitor -ts_type cn -snes_fd_color -ts_max_time 10 -snes_converged_reason" 2 3

runpattern_4:
	-@../testit.sh pattern "-da_refine 3 -ptn_call_back_report -ts_monitor" 1 4

runpattern_5:
	-@../testit.sh pattern "-da_refine 4 -ptn_call_back_report -ts_type bdf -ts_max_time 1 -snes_converged_reason -ts_monitor" 1 5

test_ode: runode_1 runode_2 runode_3

test_odejac: runodejac_1 runodejac_2

test_heat: runheat_1 runheat_2

test_pattern: runpattern_1 runpattern_2 runpattern_3 runpattern_4 runpattern_5

test: test_ode test_odejac test_heat test_pattern

# etc

.PHONY: distclean runode_1 runode_2 runode_3 runodejac_1 runodejac_2 runheat_1 runheat_2 runpattern_1 runpattern_2 runpattern_3 runpattern_4 runpattern_5 test test_ode test_odejac test_heat test_pattern

distclean:
	@rm -f *~ ode odejac heat pattern *tmp
	@rm -f *.pyc *.dat *.dat.info *.png PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/

