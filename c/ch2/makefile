include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vecmatksp: vecmatksp.o  chkopts
	-${CLINKER} -o vecmatksp vecmatksp.o  ${PETSC_KSP_LIB}
	${RM} vecmatksp.o

tri: tri.o  chkopts
	-${CLINKER} -o tri tri.o  ${PETSC_KSP_LIB}
	${RM} tri.o

# testing
PROG = vecmatksp
OPTS =
TEST = 1
PROC = 1
runvecmatksp_1:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

PROG = tri
OPTS = -ksp_monitor -a_mat_view ::ascii_dense
TEST = 1
PROC = 1
runtri_1:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

OPTS = -tri_m 2000 -ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_monitor -ksp_converged_reason
TEST = 2
PROC = 4
runtri_2:
	-@${MAKE} ${PROG} > maketmp 2>&1;
	-@${MPIEXEC} -n ${PROC} ./${PROG} ${OPTS} > tmp 2>&1; \
	  if (${DIFF} output/${PROG}.test${TEST} tmp) then true; \
	  else  printf "ERROR: ${PROC}-process test #${TEST} of ${PWD}/${PROG}; diffs above\n"; fi; \
	  ${RM} -f maketmp tmp

# etc

.PHONY: distclean

distclean:
	@rm -f *~ vecmatksp tri

