# testing

runfirefish_1:
	-@../../c/testit.sh firefish.py "-mx 5 -my 5 -order 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfirefish_2:
	-@../../c/testit.sh firefish.py "-refine 2 -s_ksp_converged_reason -s_pc_type mg" 1 2

test_firefish: runfirefish_1 runfirefish_2

test: test_firefish

# etc

.PHONY: clean runfirefish_1 runfirefish_2 test test_firefish

.PHONY: clean

clean:
	@rm -f *.pyc *.pvd *.vtu

