include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

e: e.o  chkopts
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

shuffle: shuffle.o  chkopts
	-${CLINKER} -o shuffle shuffle.o  ${PETSC_LIB}
	${RM} shuffle.o

# etc

PROG = e
TEST = 1
PROC = 1
rune_1:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} > tmp 2>&1; \
	  if (${DIFF} regress/${PROG}.test${TEST} tmp) then true; \
	  else  printf "problem with test ${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

PROG = e
TEST = 2
PROC = 4
rune_2:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} > tmp 2>&1; \
	  if (${DIFF} regress/${PROG}.test${TEST} tmp) then true; \
	  else  printf "problem with test ${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

.PHONY: distclean

distclean:
	@rm -f *~ e shuffle

