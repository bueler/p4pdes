include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ode: ode.o  chkopts
	-${CLINKER} -o ode ode.o  ${PETSC_LIB}
	${RM} ode.o

heat: heat.o  chkopts
	-${CLINKER} -o heat heat.o  ${PETSC_LIB}
	${RM} heat.o

pattern: pattern.o  chkopts
	-${CLINKER} -o pattern pattern.o  ${PETSC_LIB}
	${RM} pattern.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ ode heat pattern

