include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vecmatksp: vecmatksp.o  chkopts
	-${CLINKER} -o vecmatksp vecmatksp.o  ${PETSC_LIB}
	${RM} vecmatksp.o

tri: tri.o  chkopts
	-${CLINKER} -o tri tri.o  ${PETSC_LIB}
	${RM} tri.o

# testing
runvecmatksp_1:
	-@../testit.sh vecmatksp "" 1 1

runtri_1:
	-@../testit.sh tri "-a_mat_view ::ascii_dense" 1 1

runtri_2:
	-@../testit.sh tri "-tri_m 2000 -ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_converged_reason" 4 2

test: runvecmatksp_1 runtri_1 runtri_2

# etc

.PHONY: distclean runvecmatksp_1 runtri_1 runtri_2 test

distclean:
	@rm -f *~ vecmatksp tri

