include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

sparsemat: sparsemat.o
	-${CLINKER} -o sparsemat sparsemat.o  ${PETSC_LIB}
	${RM} sparsemat.o

vecmatksp: vecmatksp.o
	-${CLINKER} -o vecmatksp vecmatksp.o  ${PETSC_LIB}
	${RM} vecmatksp.o

tri: tri.o
	-${CLINKER} -o tri tri.o  ${PETSC_LIB}
	${RM} tri.o

loadsolve: loadsolve.o
	-${CLINKER} -o loadsolve loadsolve.o  ${PETSC_LIB}
	${RM} loadsolve.o

# testing
runsparsemat_1:
	-@../testit.sh sparsemat "-mat_view" 1 1

runvecmatksp_1:
	-@../testit.sh vecmatksp "" 1 1

runtri_1:
	-@../testit.sh tri "-a_mat_view ::ascii_dense" 1 1

runtri_2:
	-@../testit.sh tri "-tri_m 1000 -ksp_rtol 1.0e-4 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2

runloadsolve_1:
	-@./tri -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat > /dev/null
	-@../testit.sh loadsolve "-verbose -fA A.dat -fb b.dat -ksp_view_mat -ksp_view_rhs -ksp_view_solution" 1 1

test_sparsemat: runsparsemat_1

test_vecmatksp: runvecmatksp_1

test_tri: runtri_1 runtri_2

test_loadsolve: runloadsolve_1

test: test_sparsemat test_vecmatksp test_tri test_loadsolve

# etc

.PHONY: distclean runvecmatksp_1 runtri_1 runtri_2 runloadsolve_1 test test_vecmatksp test_tri test_loadsolve

distclean:
	@rm -f *~ sparsemat vecmatksp tri loadsolve *tmp
	@rm -f *.dat *.dat.info

