include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

e: e.o  chkopts
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

# testing
PROG = e
TEST = 1
PROC = 1
rune_1:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

.PHONY: distclean

distclean:
	@rm -f *~ e

