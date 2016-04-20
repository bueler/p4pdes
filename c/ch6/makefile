include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ode: ode.o  chkopts
	-${CLINKER} -o ode ode.o  ${PETSC_LIB}
	${RM} ode.o

odejac: odejac.o  chkopts
	-${CLINKER} -o odejac odejac.o  ${PETSC_LIB}
	${RM} odejac.o

heat: heat.o  chkopts
	-${CLINKER} -o heat heat.o  ${PETSC_LIB}
	${RM} heat.o

pattern: pattern.o  chkopts
	-${CLINKER} -o pattern pattern.o  ${PETSC_LIB}
	${RM} pattern.o


# testing
runode_1:
	-@../testit.sh ode "-ts_init_time 19.0 -ts_dt 0.01 -ts_final_time 20.0" 1 1

runode_2:
	-@../testit.sh ode "-ts_type cn -snes_fd -ts_final_time 1.0" 1 2

runode_3:
	-@../testit.sh ode "-ts_monitor -ts_final_time 1.0" 1 3

runodejac_1:
	-@../testit.sh odejac "-ts_final_time 1.0" 1 1


# etc

.PHONY: distclean

distclean:
	@rm -f *~ ode odejac heat pattern *.pyc *.dat *.dat.info

