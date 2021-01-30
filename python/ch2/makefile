# testing
COUTPUTS := ../../c/ch2/output/

output:
	@ln -sf $(COUTPUTS)

runsparsemat_1: output
	-@../ptestit.sh sparsemat "-mat_view" 1 1

runvecmatksp_1: output
	-@../ptestit.sh vecmatksp "" 1 1

runtri_1: output
	-@../ptestit.sh tri "-a_mat_view ::ascii_dense" 1 1

runtri_2:
	-@../ptestit.sh tri "-tri_m 1000 -ksp_rtol 1.0e-4 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2

test_sparsemat: runsparsemat_1

test_vecmatksp: runvecmatksp_1

test_tri: runtri_1 runtri_2

test: test_sparsemat test_vecmatksp test_tri

# etc

.PHONY: clean runsparsemat_1 runvecmatksp_1 runtri_1 runtri_2 test test_sparsemat test_vecmatksp test_tri

clean:
	@rm -f *.dat *.dat.info
	@rm -rf output

