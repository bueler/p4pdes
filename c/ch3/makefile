include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o chkopts
	-${CLINKER} -o poisson poisson.o ${PETSC_KSP_LIB}
	${RM} poisson.o

# testing
PROG = poisson
OPTS = -da_grid_x 5 -da_grid_y 7
TEST = 1
PROC = 1
runpoisson_1:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

OPTS = -da_grid_x 4 -da_grid_y 3 -a_mat_view
TEST = 2
PROC = 1
runpoisson_2:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

OPTS = -da_grid_x 10 -da_grid_y 10
TEST = 3
PROC = 4
runpoisson_3:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson

