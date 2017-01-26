include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vecmatksp: vecmatksp.o  chkopts
	-${CLINKER} -o vecmatksp vecmatksp.o  ${PETSC_LIB}
	${RM} vecmatksp.o

tri: tri.o  chkopts
	-${CLINKER} -o tri tri.o  ${PETSC_LIB}
	${RM} tri.o

loadsolve: loadsolve.o  chkopts
	-${CLINKER} -o loadsolve loadsolve.o  ${PETSC_LIB}
	${RM} loadsolve.o

# testing
runvecmatksp_1:
	-@../testit.sh vecmatksp "" 1 1

runtri_1:
	-@../testit.sh tri "-a_mat_view ::ascii_dense" 1 1

runtri_2:
	-@../testit.sh tri "-tri_m 1000 -ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_converged_reason" 2 2

runloadsolve_1:
	-@./tri -ksp_view_mat binary:Ab.dat -ksp_view_rhs binary:Ab.dat::append > /dev/null
	-@../testit.sh loadsolve "-notime -f Ab.dat -ksp_view_mat -ksp_view_rhs -ksp_view_solution" 1 1

test: runvecmatksp_1 runtri_1 runtri_2 runloadsolve_1

# etc

.PHONY: distclean runvecmatksp_1 runtri_1 runtri_2 test

distclean:
	@rm -f *~ vecmatksp tri

